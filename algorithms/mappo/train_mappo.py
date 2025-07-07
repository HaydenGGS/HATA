import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from algorithms.utils.util import check
from codes.algorithms.mappo.predict_mi import predict_net, predict_net_withtrait
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from env.environment import environment as env


def actions_2_Onehot(actions, onehot_size):
    max_i = np.array(actions.shape)[0]
    max_j = np.array(actions.shape)[1]
    max_k = np.array(actions.shape)[2]
    actions_onehot = np.zeros((max_i,max_j,max_k,onehot_size))
    for i in range(max_i):
        for j in range(max_j):
            for k in range(max_k):
                actions_idx = actions[i][j][k][0]
                actions_onehot[i,j,k,int(actions_idx)] = 1
    return actions_onehot

class MAPPO():
    """
    Trainer class for MAPPO to update policies.
    """

    def __init__(self, args, envs, policy, device=torch.device("cuda")):
        
        self.policy = policy
        self.args = args
        self.envs =envs
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # Mutual Information Predict Net 
        env_ = env(self.args)
        obs_dim = self.args.obs_dim
        self.eval_predict_withoutid = predict_net(
            args.hidden_size + obs_dim + env_.action_dim, 128, obs_dim, False)
        self.target_predict_withoutid = predict_net(
            args.hidden_size + obs_dim + env_.action_dim, 128, obs_dim, False)

        self.eval_predict_withid = predict_net_withtrait(args.hidden_size + obs_dim + env_.action_dim + self.args.trait_dim, 128,
                                                            obs_dim, False)
        self.target_predict_withid = predict_net_withtrait(args.hidden_size + obs_dim + env_.action_dim + self.args.trait_dim, 128,
                                                            obs_dim, False)
        
        self.eval_predict_withid.to(device)
        self.target_predict_withid.to(device)
        self.eval_predict_withoutid.to(device)
        self.target_predict_withoutid.to(device)
        
        self.loss_withid = None
        self.loss_withoutid = None
        self.mi_loss = 0
        self.backflag = 1

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, trait_loss, trait_weight, update_actor=True):
        """
        Update actor and critic networks.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              trait_weight,
                                                                              available_actions_batch,
                                                                              active_masks_batch)

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss 
        self.mi_loss = (self.loss_withid + self.loss_withoutid)
        trait_loss = trait_loss.to(self.device)
        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward(retain_graph = True)
            # if self.backflag == 1:
            #     (policy_loss + self.mi_loss - dist_entropy * self.entropy_coef).backward(retain_graph = True)
            #     self.backflag = 0
            # elif self.backflag == 0:
            #     (policy_loss + self.mi_loss - dist_entropy * self.entropy_coef).backward()
        
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        ((value_loss + (self.mi_loss*0.2)) * self.value_loss_coef).backward(retain_graph = True)

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def predictnet_update_(self):
        self.target_predict_withid.load_state_dict(
            self.eval_predict_withid.state_dict())
        self.target_predict_withoutid.load_state_dict(
            self.eval_predict_withoutid.state_dict())
        
    def train(self, buffer, step, trait_feature, trait_loss, trait_weight, sr_last, update_actor=True):
        """
        Perform a training update using minibatch GD.
        """
        # mutual information as rewards
        actions = buffer.actions
        available_actions = buffer.available_actions 
        action_log_probs = buffer.action_log_probs 
        active_masks = buffer.active_masks 
        obs = buffer.obs
        returns = buffer.returns 
        rnn_states = buffer.rnn_states 
        rnn_hidden = self.policy.rnn_hidden 
        onehot_size = np.array(available_actions.shape)[-1]
        actions_onthot = actions_2_Onehot(actions, onehot_size) 
    
        obs = torch.from_numpy(buffer.obs).permute(1,0,2,3)[:,:-1] 
        obs_next = torch.from_numpy(buffer.obs).permute(1,0,2,3)[:,1:] 
        rnn_states = torch.from_numpy(buffer.rnn_states).squeeze().permute(1,0,2,3) 
        actions_onthot = torch.from_numpy(actions_onthot).permute(1,0,2,3)
        action_log_probs = torch.from_numpy(action_log_probs).permute(1,0,2,3) 
        masks_clone = torch.from_numpy(active_masks).detach().clone().permute(1,0,2,3) 

        masks_clone = masks_clone.reshape(-1, masks_clone.shape[-1]).to(self.device)
        obs_intrinsic = obs.clone().permute(0,2,1,3)
        obs_intrinsic = obs_intrinsic.reshape(
            -1, obs_intrinsic.shape[-2], obs_intrinsic.shape[-1]).to(self.device)

        eval_h_intrinsic = rnn_states.clone().permute(0,2,1,3)
        eval_h_intrinsic = eval_h_intrinsic.reshape(
            -1, eval_h_intrinsic.shape[-2], eval_h_intrinsic.shape[-1]).to(self.device)

        h_cat = torch.cat([rnn_hidden, eval_h_intrinsic[:, :-2]], dim=1)
        trait_features = trait_feature.to(self.device).reshape(-1, self.args.num_agent, self.args.trait_dim).expand([self.args.episode_length, self.args.n_rollout_threads, self.args.num_agent, self.args.trait_dim]).permute(1, 2, 0, 3)

        actions_onehot_clone = actions_onthot.clone().permute(0, 2, 1, 3).to(self.device)
        action_log_probs = action_log_probs.permute(0, 2, 1, 3).to(self.device)
        intrinsic_input_1 = torch.cat(
            [h_cat, obs_intrinsic, actions_onehot_clone.reshape(-1, actions_onehot_clone.shape[-2], actions_onehot_clone.shape[-1])], dim=-1)
        intrinsic_input_2 = torch.cat(
            [intrinsic_input_1, trait_features.reshape(-1, trait_features.shape[-2], trait_features.shape[-1])], dim=-1)
        intrinsic_input_1 = intrinsic_input_1.reshape(-1, intrinsic_input_1.shape[-1]).to(torch.float32)# 1200 * 83
        intrinsic_input_2 = intrinsic_input_2.reshape(-1, intrinsic_input_2.shape[-1]).to(torch.float32)# 1200 * 85

        next_obs_intrinsic = obs_next.clone().permute(0, 2, 1, 3)
        next_obs_intrinsic = next_obs_intrinsic.reshape(
            -1, next_obs_intrinsic.shape[-2], next_obs_intrinsic.shape[-1])
        next_obs_intrinsic = next_obs_intrinsic.reshape(
            -1, next_obs_intrinsic.shape[-1]).to(self.device)

        log_p_o = self.target_predict_withoutid.get_log_pi(
            intrinsic_input_1, next_obs_intrinsic)
        log_q_o = self.target_predict_withid.get_log_pi(
            intrinsic_input_2, next_obs_intrinsic, trait_features.reshape([-1, trait_features.shape[-1]]))

        mean_p = action_log_probs.mean(dim=1) 
        pi_diverge = torch.cat(
            [(action_log_probs[:,id] * torch.log(action_log_probs[:,id] / mean_p)).sum(
                dim=-1, keepdim=True) for id in range(self.args.num_agent)], dim=-1).permute(0,2,1).unsqueeze(-1)
        
        intrinsic_rewards = 0.5 * log_q_o - log_p_o
        intrinsic_rewards = intrinsic_rewards.reshape(
            -1, obs_intrinsic.shape[1], intrinsic_rewards.shape[-1])
        intrinsic_rewards = intrinsic_rewards.reshape(
            -1, obs.shape[2], obs_intrinsic.shape[1], intrinsic_rewards.shape[-1])

        intrinsic_rewards = intrinsic_rewards + 0.5 * pi_diverge
        
        if self.args.anneal:
            intrinsic_rewards = self.args.anneal * max(1 - max(sr_last, step/10000), 0.1) * intrinsic_rewards

        # update predict network
        trait_features = trait_features.reshape([-1, trait_features.shape[-1]])
        for index in BatchSampler(SubsetRandomSampler(range(intrinsic_input_1.shape[0])), 256, False):
            self.loss_withid = self.eval_predict_withoutid.update(
                intrinsic_input_1[index], next_obs_intrinsic[index], masks_clone[index])
            self.loss_withoutid = self.eval_predict_withid.update(
                intrinsic_input_2[index], next_obs_intrinsic[index], trait_features[index], masks_clone[index])
            
        returns[:-1] = returns[:-1] + ( 0.1 * intrinsic_rewards.permute(2,0,1,3).cpu().detach().numpy() )

        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['loss_withid'] = 0
        train_info['loss_withoutid'] = 0
        train_info['mi_loss'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, trait_loss, trait_weight, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        # predict_net update
        train_info['loss_withid'] += self.loss_withid
        train_info['loss_withoutid'] += self.loss_withoutid
        train_info['mi_loss'] += self.mi_loss
        self.predictnet_update_()

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()


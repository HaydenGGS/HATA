import time
import os
import numpy as np
import torch
from runner.base_runner_share import Runner, RunningStat

def _t2n(x):
    return x.detach().cpu().numpy()

class EnvRunner(Runner):

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.config = config
        self.actions = None 
        self.action = None   
        self.running_stat = RunningStat((1,))

        self.success_rate = torch.zeros((self.args.n_rollout_threads))
        self.cap_frequency = torch.zeros((self.args.n_rollout_threads))
        self.envs = config['envs']
        self.args = config['all_args']
        self.a1 = self.args.weight_dn
        self.eval_div = self.args.div_sr_fre
        self.num_stage_runner = torch.zeros((self.args.n_rollout_threads))
        self.cl_count = torch.zeros((self.args.n_rollout_threads))
        self.stage_cl_count = self.args.stage_cl_count


    def run(self):
        self.reset_run()
        self.start = time.time()
        self.time_now = time.strftime('%m%d-%H-%M', time.localtime(time.time()))
        print(f"start training: {self.time_now}")
        
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        count_success = torch.zeros((self.args.n_rollout_threads))
        total_success_count = torch.zeros((self.args.n_rollout_threads))
        total_fre_count = torch.zeros((self.args.n_rollout_threads))
        fre_total = torch.zeros((self.args.n_rollout_threads))
        sr_last = torch.zeros((self.args.n_rollout_threads))
        fre_last = torch.zeros((self.args.n_rollout_threads))
        trait_loss = 0
        for episode in range(1, episodes + 1):

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            _, self.r, self.v = self.envs.reset(self.num_stage_runner)
        
            """  Create Trait Map  """
            self.trait_map = torch.cat([ torch.Tensor(np.vstack((self.r[i,:],self.v[i,:]))) for i in range(self.args.n_rollout_threads)])
            assert self.args.use_double_net == True or self.args.update_single_net != 0,"no set net! use single or double network! "
            assert self.args.use_double_net == False or self.args.update_single_net == 0,"net set conflict! use single or double network!" 

            """  Single Network fixed update steps  """
            if self.args.update_single_net and episode % self.args.update_single_net != 0 :
                for name, param in self.trait_encoder.named_parameters():
                    param.requires_grad = False
            
            elif self.args.update_single_net and episode % self.args.update_single_net == 0 :
                for name, param in self.trait_encoder.named_parameters():
                    param.requires_grad = True

            """  Use map  """
            self.trait_feature = self.trait_encoder.forward(self.trait_map)

            """  Double Network fixed update steps  """
            if episode % self.args.update_double_net == 0:
                self.trait_encoder_target.load_state_dict(self.trait_encoder.state_dict())
            if self.args.use_double_net:
                self.trait_feature_target = self.trait_encoder_target.forward(self.trait_map)
                trait_loss = self.trait_encoder.update_network(self.trait_feature_target, self.trait_feature)

            self.trait_feature_all = self.a1 * self.trait_feature + (1 - self.a1) * self.trait_feature_target
            
            episode_return = np.zeros(self.n_rollout_threads) 
            last_dones = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int64)
            rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float64)
            suc_flag = np.zeros((self.n_rollout_threads, 1), dtype=np.int64)
            step_last = np.ones((self.n_rollout_threads, 1), dtype=np.int64) * self.episode_length

            for step in range(self.episode_length):
                # Sample actions
                ( values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env ) = self.collect(step)

                obs, rewards, dones, infos = self.envs.step(actions_env)
                rewards = rewards.reshape(self.n_rollout_threads, self.num_agents, 1)
                dones = dones.squeeze()

                episode_return += np.ravel(np.sum(rewards.squeeze(),axis=1))

                data = ( obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, )
                self.insert(data)

                for i in range(self.n_rollout_threads):
                    if np.all(np.ravel(dones[i, :])):
                        if suc_flag[i,:] != 1:
                            step_last[i,:] = step
                            suc_flag[i,:] = 1

            # compute return and update network
            print(f"sr_last: {np.ravel(sr_last)}, fre_last: {np.ravel(fre_last)}")
            self.compute()
            train_infos = self.train(episode, self.trait_feature_all, trait_loss, self.trait_weight_all, sr_last)
        
            total_success_count += np.ravel(suc_flag)
            total_fre_count += np.ravel(step_last)

            if episode % self.eval_div == 0 and ( episode != 0 ):
                cl_sr = total_success_count / self.eval_div
                cl_fre = (torch.ones(self.args.n_rollout_threads) * self.args.episode_length * self.eval_div - total_fre_count) / (self.eval_div * self.args.episode_length)
                
                for i in range(self.args.n_rollout_threads):
                    if cl_sr[i] > 0.9 and cl_fre[i] > 0.6:
                        self.cl_count[i] += 1
                    else:
                        self.cl_count[i] = 0

                    if self.cl_count[i] >= self.stage_cl_count:
                        self.num_stage_runner[i] += 1
                        if self.num_stage_runner[i] > self.args.stage_cl_total:
                            self.num_stage_runner[i] = self.args.stage_cl_total

                total_success_count = torch.zeros((self.args.n_rollout_threads))
                total_fre_count = torch.zeros((self.args.n_rollout_threads))
                print(f"cl_stage: {self.num_stage_runner}")

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            self.writter.add_scalar('value_loss:', train_infos['value_loss'], episode)
            self.writter.add_scalar('policy_loss:', train_infos['policy_loss'], episode)
            self.writter.add_scalar('loss_withid:', train_infos['loss_withid'], episode)
            self.writter.add_scalar('loss_withoutid:', train_infos['loss_withoutid'], episode)
            self.writter.add_scalar('mi_loss:', train_infos['mi_loss'], episode)
            self.writter.add_scalar('reward0:', episode_return[0] / self.args.reward_scale, episode)
            self.writter.add_scalar('reward1:', episode_return[1] / self.args.reward_scale, episode)
            print(f"reward:{np.ravel(episode_return / self.args.reward_scale)}, dones:{np.ravel(dones)}, episode:{episode}")

            count_success = count_success + np.ravel(suc_flag)
            step_ = np.ones((self.n_rollout_threads, 1)) * self.episode_length
            fre_total = fre_total + np.ravel((step_ - step_last) / self.episode_length)

            if episode % self.eval_div == 0 and ( episode != 0 ):
                # success rate
                success_rate = count_success / (self.eval_div)
                self.writter.add_scalar('success_rate0:', np.ravel(success_rate)[0], episode)
                self.writter.add_scalar('success_rate1:', np.ravel(success_rate)[1], episode)
                sr_last = success_rate
                suc_flag = torch.zeros((self.args.n_rollout_threads))
                # catch frequency
                cap_frequency = fre_total / (self.eval_div)
                self.writter.add_scalar('cap_frequency0:', np.ravel(cap_frequency)[0], episode)
                self.writter.add_scalar('cap_frequency1:', np.ravel(cap_frequency)[1], episode)
                fre_last = cap_frequency

                count_success = torch.zeros((self.args.n_rollout_threads))
                fre_total = torch.zeros((self.args.n_rollout_threads))
                self.end = time.time()
                print("sr: {}, frequency: {}, time: {:.2f} min".format(np.ravel(success_rate), np.ravel(cap_frequency), (self.end-self.start)/3600))
                
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps, episode)
            # save
            if episode % self.save_interval == 0 and episode != 0:
                self.save_model(episode)

    def reset_run(self):
        # reset env
        obs, r_, v_, = self.envs.reset(self.num_stage_runner) 

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1) 
            share_obs = np.expand_dims(share_obs, 1).repeat(
                self.num_agents, axis=1
            )  
        else:
            share_obs = obs
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

        return obs, r_, v_

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            self.trait_weight_all,
        )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        ) 

        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            # TODO 
            # TODO Here, you can change the shape of actions_env to fit your environment
            actions_env = actions


        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        ( obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )
    
    @torch.no_grad()
    def save_model(self, episode):
        
        self.save_model_dir = self.save_dir + '/' + self.time_now + '/' + str(episode)
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_model_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_model_dir) + "/critic.pt")
        
        
    @torch.no_grad()
    def eval(self, total_num_steps, episode):
        n_stage = np.array(2)
        count_success = 0
        count_fre = 0
        reward_total = 0
        for eval_episode in range(self.args.eval_episodes):

            eval_episode_rewards = 0
            eval_obs, _, _ = self.eval_envs.reset(n_stage)
            eval_obs = eval_obs.reshape(self.n_eval_rollout_threads, eval_obs.shape[0], eval_obs.shape[1])
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32,)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            suc_flag_eval = 0

            for eval_step in range(self.episode_length):
                self.trainer.prep_rollout()
                
                eval_action, eval_rnn_states = self.trainer.policy.act(
                    np.concatenate(eval_obs),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    deterministic=True,
                )
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

                if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.eval_envs.action_space[0].shape):
                        eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[
                            eval_actions[:, :, i]
                        ]
                        if i == 0:
                            eval_actions_env = eval_uc_actions_env
                        else:
                            eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
                elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                    eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
                else:
                    raise NotImplementedError

                eval_actions_env = eval_actions_env.squeeze()
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                eval_obs = eval_obs.reshape(self.n_eval_rollout_threads, eval_obs.shape[0], eval_obs.shape[1])
                eval_episode_rewards += np.ravel(eval_rewards)
                eval_rnn_states[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

                if np.any(np.ravel(eval_dones)):
                    if suc_flag_eval != 1:
                        print(f"success!: {eval_step}")
                        suc_flag_eval = 1
                        count_fre += self.episode_length - eval_step
                        count_success += 1


            reward_total += eval_episode_rewards.mean() / self.args.reward_scale

        print("reward_eval: {}, sr_eval: {:.3f}, fre_eval: {:.3f}".format(
            reward_total / self.args.eval_episodes, 
            count_success / self.args.eval_episodes, 
            count_fre / (self.args.eval_episodes * self.episode_length)))
        self.writter.add_scalar('reward_eval:', reward_total / self.args.eval_episodes, episode)
        self.writter.add_scalar('sr_eval:', count_success / self.args.eval_episodes, episode)
        self.writter.add_scalar('fre_eval:', count_fre / (self.args.eval_episodes * self.episode_length), episode)

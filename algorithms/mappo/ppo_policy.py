import torch
from codes.algorithms.mappo.actor_critic import Actor, Critic
from utils.util import update_linear_schedule


class MAPPO_Policy:
    """
    MAPPO Policy class.
    """
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cuda")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.args = args

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        if self.args.use_obs_atten:
            self.critic = Critic(args, self.obs_space, self.device)
        else:
            self.critic = Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.rnn_hidden = None

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, trait_w, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 trait_w,
                                                                 available_actions,
                                                                 deterministic)
        self.rnn_hidden = rnn_states_actor

        if self.args.use_obs_atten:
            values, rnn_states_critic = self.critic(obs, rnn_states_critic, actions, masks, trait_w)
        else:
            values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, actions, masks, trait_w)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, actions, masks, traits):
        """
        Get value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, actions, masks, traits)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, trait_w,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
    
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     trait_w,
                                                                     available_actions,
                                                                     active_masks)
        if self.args.use_obs_atten:
            values, _ = self.critic(obs, rnn_states_critic, action, masks, trait_w)
        else:
            values, _ = self.critic(cent_obs, rnn_states_critic, action, masks, trait_w)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, trait_w, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, trait_w, available_actions, deterministic)
        return actions, rnn_states_actor

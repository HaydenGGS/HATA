import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer
from algorithms.mappo.trait_encoder import Trait_Map

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class RunningStat:
    def __init__(self, shape):
        self.n = 0

        self.mean = np.zeros(shape)
        self.s = np.zeros(shape)
        self.std = np.zeros(shape)

    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.s = self.s + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.s / (self.n - 1) if self.n > 1 else np.square(self.mean))


class Runner(object):
    """
    Base class for training recurrent policies.
    """
    def __init__(self, config):
        self.args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agent']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.args.env_name
        self.algorithm_name = self.args.algorithm_name
        self.experiment_name = self.args.experiment_name
        # global param
        self.use_centralized_V = self.args.use_centralized_V
        self.use_obs_instead_of_state = self.args.use_obs_instead_of_state
        self.num_env_steps = self.args.num_env_steps
        self.episode_length = self.args.episode_length
        self.n_rollout_threads = self.args.n_rollout_threads
        self.n_eval_rollout_threads = self.args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.args.n_render_rollout_threads
        self.use_linear_lr_decay = self.args.use_linear_lr_decay
        self.hidden_size = self.args.hidden_size
        self.use_render = self.args.use_render
        self.recurrent_N = self.args.recurrent_N
        self.trait_feature_all = None
        self.trait_weight_all = None

        # interval
        self.save_interval = self.args.save_interval
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.log_interval = self.args.log_interval

        # dir
        self.model_dir = self.args.model_dir

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir + "/logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir + "/models")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        from codes.algorithms.mappo.train_mappo import MAPPO as TrainAlgo
        from algorithms.mappo.ppo_policy import MAPPO_Policy as Policy
        
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        # policy network init
        self.policy = Policy(self.args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.args, self.envs, self.policy, device = self.device)
        if self.args.update_single_net > 0:
            pass
        else:
            self.trait_encoder = Trait_Map(self.args, device=self.device)
            if self.args.use_double_net:
                self.trait_encoder_target = Trait_Map(self.args, device=self.device)
        
        # buffer 
        self.buffer = SharedReplayBuffer(self.args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.args.use_obs_atten:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.obs[-1]),
                                                    np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                    np.concatenate(self.buffer.actions[-1]),
                                                    np.concatenate(self.buffer.masks[-1]),
                                                    self.trait_weight_all)
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                    np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                    np.concatenate(self.buffer.actions[-1]),
                                                    np.concatenate(self.buffer.masks[-1]),
                                                    self.trait_weight_all)
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self, episode, feature, trait_loss, trait_weight, sr_last):

        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer, episode, feature, trait_loss, trait_weight, sr_last)      
        self.buffer.after_update()

        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        """
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
import torch
from torch import nn
import numpy as np


# actions change from num to one-hot form
def actions_2_Onehot(actions, onehot_size = 5):
    n_threads = actions.shape[0]
    actions_onehot = np.zeros((n_threads,onehot_size))
    for i in range(n_threads):
        actions_idx = actions[i][0]
        actions_onehot[i,int(actions_idx)] = 1
    return actions_onehot

class Feature_Process(nn.Module):
    # process observation feature
    def __init__(self, args, obs_shape, hidden_size, device=torch.device("cuda")):
        super(Feature_Process, self).__init__()
        # args
        self.args = args
        self.hidden_size = hidden_size
        self.obs_shape = obs_shape[0]

        # agent indivadal state
        self.state_agent = self.args.obs_dim
        self.device = device
        self.layer1 = nn.Linear(self.state_agent, self.hidden_size*2)
        self.layer2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.layer3 = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.layer_action = nn.Linear(5, self.hidden_size)
        self.layer_o_a = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.args.use_obs_atten_actor:
            self.encoder_interaction = Trait_Attention_Actor(args, ((self.args.num_agent - 1) * 2), device=self.device,)
        else:
            self.encoder_interaction = nn.Sequential(
                nn.Linear(((self.args.num_agent - 1) * 2), self.hidden_size * 2),
                nn.ReLU(),
                )

    def forward(self, obs, trait_w):
        
        state_position = obs
        x = torch.relu(self.layer1(state_position))  # 6,64
        x_ = torch.relu(self.layer2(x))

        all_feature = torch.relu(self.layer3(x_))
        
        return all_feature
    
    def reset(self):
        self.encoder_inter.hidden = None
    
    @property
    def output_size(self):
        output_size = self.hidden_size
        return output_size


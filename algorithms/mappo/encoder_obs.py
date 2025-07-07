import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from algorithms.utils.gnn.graph_conv_module import GraphConvolutionModule

"""
ResBlock
"""
class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block1 = nn.Conv1d(channel, channel, 3, 1, 1)
        self.block2 = nn.Conv1d(channel, channel, 3, 1, 1)
    def forward(self, x):
        x_res = x
        x = self.block1(x)
        x = F.relu(x)
        x = self.block2(x)
        x += x_res
        x = F.relu(x)
        
        return x

"""
Multi Head Attention
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, args, input_dim, output_dim, num_heads, device=torch.device("cuda")):
        super().__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.args = args

        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)

    def forward(self, input, index, trait_weight, attn_mask):
        batch_size, num_agents, input_dim = input.size()
        assert input_dim == self.input_dim
        
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  
        k_s = self.W_K(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  
        v_s = self.W_V(input).view(batch_size, num_agents, self.num_heads, -1).transpose(1,2)  

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        assert attn_mask.size(0) == batch_size, 'mask dim {} while batch size {}'.format(attn_mask.size(0), batch_size)

        attn_mask = attn_mask.unsqueeze(1).repeat_interleave(self.num_heads, 1).to(self.device) 
        assert attn_mask.size() == (batch_size, self.num_heads, num_agents, num_agents)

        with autocast(enabled=False):
            scores = torch.matmul(q_s.float(), k_s.float().transpose(-1, -2)) / (self.output_dim**0.5) # scores 
            scores.masked_fill_(attn_mask, -1e9) 
            attn = F.softmax(scores, dim=-1).to(self.device)
        
        weight = attn
        context = torch.matmul(weight, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_agents, self.num_heads*self.output_dim) 
        output = self.W_O(context)

        return output 
    
"""
Communication 
"""
class CommBlock(nn.Module):
    def __init__(self, args, input_dim, output_dim=64, num_heads=4, num_layers=2, device=torch.device("cuda")):
        super(CommBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        self.args = args
        
        self.atten = MultiHeadAttention(self.args, input_dim, output_dim, num_heads)
        
        self.update_cell = nn.GRUCell(output_dim, input_dim)

    def forward(self, latent, comm_mask, trait_weight):
        """
        latent shape: batch_size x num_agents x hidden_size
        """
        num_agents = latent.size(1)
        n_threads = latent.size(0)

        update_mask = comm_mask.sum(dim=-1) > 1
        comm_idx = update_mask.nonzero(as_tuple=True)
        if len(comm_idx[0]) == 0:
            return latent
        if len(comm_idx)>1:
            update_mask = update_mask.unsqueeze(2)
        attn_mask = comm_mask==False

        for index in range(self.num_layers):
            info = self.atten(latent, index, trait_weight, attn_mask=attn_mask)

            if len(comm_idx)==1:
                batch_idx = torch.zeros(len(comm_idx[0]), dtype=torch.long)
                latent[batch_idx, comm_idx[0]] = self.update_cell(info[batch_idx, comm_idx[0]], latent[batch_idx, comm_idx[0]])
            else:
                update_info = self.update_cell(info.contiguous().view(-1, self.output_dim), latent.contiguous().view(-1, self.input_dim)).view(n_threads, num_agents, self.input_dim)

                update_mask = update_mask.to(self.device)
                latent = torch.where(update_mask, update_info, latent)

        return latent
        
"""
Encoder Attention
"""
class Trait_Attention(nn.Module):

    def __init__(self, args, obs_shape, device=torch.device("cuda"), use_rnn=False, use_action=False):
        super(Trait_Attention, self).__init__()
        self.args = args
        self.n_action = 1
        self.state_shape = obs_shape
        actor_input_shape = self.state_shape
        self.use_rnn = use_rnn
        self.use_action = use_action
        self.device = device

        self.hidden_size = self.args.hidden_size
        self.encoding = nn.Sequential(
            nn.Linear(actor_input_shape, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, self.hidden_size*2),
            nn.ReLU(),
        )
        self.hidden = None
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.comm = CommBlock(self.args, self.hidden_size*2)

        self.decoder = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)

    @torch.no_grad()
    def forward(self, obs, hidden_state=None, actions=None, trait_weight=None):

        n_agent = self.args.num_agent
        input_all = obs.reshape(-1, n_agent, self.state_shape)

        obs_encoding = self.encoding(input_all) # 2,4,128

        self.hidden = obs_encoding
        n = self.hidden.shape[0]

        comm_adj = torch.ones((n, n_agent, n_agent))
        for i in range(n_agent):
            comm_adj[:,i,i] = 0
        self.hidden = self.comm(self.hidden, comm_adj, trait_weight)

        self.hidden = self.hidden.reshape(-1, self.hidden_size * 2)

        return self.hidden
    
    def reset(self):
        self.hidden = None
    
    @property
    def output_size(self):
        output_size = self.hidden_size
        return output_size

    
"""
Encoder Attention1
"""
class Trait_Attention1(nn.Module):

    def __init__(self, args, obs_shape, device=torch.device("cuda"), use_rnn=False, use_action=False):
        super(Trait_Attention1, self).__init__()
        self.args = args
        self.n_action = 1
        self.state_shape = obs_shape
        critic_input_shape = self.state_shape
        self.use_rnn = use_rnn
        self.use_action = use_action
        self.device = device

        self.encoding_obs = nn.Linear(obs_shape, self.args.hidden_size)

        self.hidden_size = self.args.hidden_size
        self.encoding = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size*2, 3, 1, 1),
            nn.ReLU(True),
            ResBlock(self.hidden_size*2),
            ResBlock(self.hidden_size*2),
            nn.ReLU(True),
            nn.Conv1d(self.hidden_size*2, self.hidden_size, 3, 1, 1),
            nn.ReLU(True),
        )

        self.gat_hid_size = 64
        self.gat_out_size = 32
        self.sub_processor1 = GraphConvolutionModule(self.hidden_size, self.gat_hid_size)
        self.sub_processor2 = GraphConvolutionModule(self.gat_hid_size, self.hidden_size)
        self.gat_encoder = GraphConvolutionModule(self.hidden_size, self.gat_out_size)

        self.sub_scheduler_mlp1 = nn.Sequential(
            nn.Linear(self.gat_out_size*2, self.gat_out_size//2),
            nn.ReLU(),
            nn.Linear(self.gat_out_size//2, self.gat_out_size//2),
            nn.ReLU(),
            nn.Linear(self.gat_out_size//2, 2))
    
        self.sub_scheduler_mlp2 = nn.Sequential(
            nn.Linear(self.gat_out_size*2, self.gat_out_size//2),
            nn.ReLU(),
            nn.Linear(self.gat_out_size//2, self.gat_out_size//2),
            nn.ReLU(),
            nn.Linear(self.gat_out_size//2, 2))


        self.hidden = None
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.comm = CommBlock(self.args, self.hidden_size)

        self.decoder = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)

    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, directed=False):

        n = self.args.num_agent
        batch_size = hidden_state.size(0)
        hid_size = hidden_state.size(-1)

        hard_attn_input = torch.cat([hidden_state.repeat(1, 1, n).view(batch_size, n * n, -1), hidden_state.repeat(1, n, 1)], dim=1).view(batch_size, n, -1, 2 * hid_size)
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True) 
        else:
            hard_attn_output = F.gumbel_softmax(0.5*sub_scheduler_mlp(hard_attn_input)+0.5*sub_scheduler_mlp(hard_attn_input.permute(1,0,2)), hard=True)

        hard_attn_output = torch.narrow(hard_attn_output, 3, 1, 1) 
        adj = hard_attn_output.squeeze() 
        
        return adj
    
    @torch.no_grad()
    def forward(self, obs, comm_adj, hidden_state=None, actions=None, trait_weight=None):

        n_agent = self.args.num_agent
        obs = obs.view(-1, n_agent, self.hidden_size)
        n = obs.shape[0]

        comm_adj = torch.ones((n, n_agent, n_agent))
        for i in range(n_agent):
            comm_adj[:,i,i] = 1

        encoder_state = self.gat_encoder(obs, comm_adj)
        adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoder_state, True)
        obs = F.elu(self.sub_processor1(obs, adj1))

        self.hidden = obs
        self.hidden = self.comm(self.hidden, adj1, trait_weight)

        self.hidden = self.hidden.reshape(-1, self.hidden_size)

        return self.hidden
    
    def reset(self):
        self.hidden = None
    
    @property
    def output_size(self):
        output_size = self.hidden_size
        return output_size



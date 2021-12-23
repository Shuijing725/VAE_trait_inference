import torch.nn as nn
import torch
import numpy as np

from rl.utils import init
from rl.rl_models.rnn_base import RNNBase, reshapeT


'''
LSTM without attention policy network, for ablation study
'''
class LSTM_NO_ATTN(nn.Module):
    def __init__(self, obs_space_dict, config, task):
        super(LSTM_NO_ATTN, self).__init__()
        self.config = config
        self.is_recurrent = True
        self.task = task

        self.human_num = obs_space_dict['spatial_edges'].shape[0]

        self.seq_length = config.ppo.num_steps
        self.nenv = config.training.num_processes
        self.nminibatch = config.ppo.num_mini_batch

        self.output_size = config.network.rnn_hidden_size

        # init the parameters
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))


        self.actor = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh(),
            init_(nn.Linear(self.output_size, self.output_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(self.output_size, 1))


        # robot node size + robot temporal edge feature size + (one) robot spatial edge feature size
        # driving: 2 + 2 = 4
        self.robot_state_size = obs_space_dict['robot_node'].shape[1] + obs_space_dict['temporal_edges'].shape[1]
        # ego car state size + one other car's state size: 4+ 4 = 8
        input_size = self.robot_state_size + self.config.network.human_state_input_size

        self.mlp1 = nn.Sequential(
            init_(nn.Linear(input_size, 150)), nn.ReLU(),
            init_(nn.Linear(150, 100)), nn.ReLU())

        self.mlp2 = nn.Sequential(
            init_(nn.Linear(100, 100)), nn.ReLU(),
            init_(nn.Linear(100, 50)), nn.ReLU())

        self.mlp3 = nn.Sequential(
            init_(nn.Linear(50+self.robot_state_size, 150)), nn.ReLU(),
            init_(nn.Linear(150, 100)), nn.ReLU(),
            init_(nn.Linear(100, config.network.embedding_size)), nn.ReLU())

        self.RNN = RNNBase(config)

        self.train()

    '''
    forward part for rl
    returns: output of rnn & new hidden state of rnn
    '''
    def _forward(self, inputs, rnn_hxs, masks, seq_length, nenv):
        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)  # seq_length, nenv, 1, 2/7
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)  # seq_length, nenv, 1, 2
        humans_obs = reshapeT(inputs['spatial_edges'], seq_length, nenv)  # seq_length, nenv, human_num, 3/2
        hidden_states = reshapeT(rnn_hxs['rnn'], 1, nenv)  # 1, nenv, 1, 256
        masks = reshapeT(masks, seq_length, nenv)  # seq_length, nenv, 1

        # [seq_length, nenv, 1, 2/7]+[seq_length, nenv, 1, 2] = [seq_length, nenv, 1, 4/9]
        robot_obs = torch.cat((robot_node, temporal_edges), dim=-1)

        # print(human_num)
        # seq_length, nenv, human_num, 4/9
        robot_obs_tile = robot_obs.expand([seq_length, nenv, self.human_num, self.robot_state_size])
        # [seq_length, nenv, human_num, 4] + [seq_length, nenv, human_num, 4] = [seq_length, nenv, human_num, 8]
        XX = torch.cat((robot_obs_tile, humans_obs), dim=-1)

        # mlp1
        # [seq_length, nenv, human_num, 100]
        s = self.mlp1(XX)

        # mlp2
        s = self.mlp2(s)  # [seq_length, nenv, human_num, 50]
        # sum along the human_num axis
        s = torch.sum(s, dim=-2, keepdim=True)
        # [seq_len, nenv, 1, 50] + [seq_len, nenv, 1, 4] = [seq_len, nenv, 1, 54]
        joint_state = torch.cat([s, robot_obs], dim=-1)

        # mlp3
        rnn_in = self.mlp3(joint_state)  # [seq_length, nenv, 1, 64]


        # lstm
        # outputs: x: [seq_length, nenv, 12, 256] h_new: [1, nenv, 12, 256]
        x, h_new = self.RNN._forward_gru(rnn_in, hidden_states, masks)
        x = x[:, :, 0, :]  # [seq_length, nenv, 256]
        return x, h_new

    '''
    forward function for rl: returns critic output, actor features, and new rnn hidden state
    '''
    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test time
            seq_length = 1
            nenv = self.nenv # 12

        else:
            # Training time
            seq_length = self.seq_length # 30
            nenv = self.nenv // self.nminibatch # 12/2 = 6

        x, h_new = self._forward(inputs, rnn_hxs, masks, seq_length, nenv)
        rnn_hxs['rnn'] = h_new.squeeze(0)  # [nenv, 1, 256]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs


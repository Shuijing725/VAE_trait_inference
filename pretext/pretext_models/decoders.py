import numpy as np
import torch
import torch.nn as nn

from rl.utils import init


'''
GRU decoder network for trait prediction pretext task, 
only used in our method, not Morton et al
'''
class LSTM_DECODER(nn.Module):
    def __init__(self, obs_space_dict, config, task, latent_size):
        super(LSTM_DECODER, self).__init__()
        self.config = config
        self.is_recurrent = True
        self.task = task
        assert task in ['pretext_predict', 'rl_predict']

        self.human_num = obs_space_dict['spatial_edges'].shape[0]
        # number of edges per agent = total number of spatial edges / total number of nodes
        self.spatial_edge_num = int(self.human_num // obs_space_dict['pretext_nodes'].shape[0])


        self.seq_length = config.pretext.num_steps
        self.nenv = config.pretext.batch_size
        self.nminibatch = 1
        self.output_size = config.network.rnn_hidden_size

        # init the parameters
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        # size of decoder output 1 + 1 + 2 = 4
        self.state_action_size = 2
        input_size = self.state_action_size + latent_size

        self.mlp_in = nn.Sequential(
            init_(nn.Linear(input_size, 32)), nn.ReLU(),
            init_(nn.Linear(32, config.network.embedding_size)), nn.ReLU())

        # state-action pair's dimension = 1, range is in [-1, 1]
        self.output_linear = nn.Sequential(
            init_(nn.Linear(self.output_size, self.output_size//2)), nn.ReLU(),
            init_(nn.Linear(self.output_size//2, self.state_action_size)), nn.Tanh())

        self.RNN = nn.GRU(config.network.embedding_size, config.network.rnn_hidden_size)
        # initialize rnn
        for name, param in self.RNN.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.train()

    # rnn_hxs should be zeros!
    # z: nenv, seq_len, 2
    def _forward(self, z, rnn_hxs, seq_length, nenv):
        z = z.permute(1, 0, 2)  # seq_len, nenv, 2
        # a special state to indicate the start of trajectory
        if self.config.training.no_cuda:
            state = -2 * torch.ones(nenv, self.state_action_size).cpu()
            outputs = torch.zeros(seq_length, nenv, self.state_action_size).cpu()
        else:
            state = -2 * torch.ones(nenv, self.state_action_size).cuda()
            outputs = torch.zeros(seq_length, nenv, self.state_action_size).cuda()

        hidden_states = rnn_hxs['rnn'].unsqueeze(0)
        # the initial hidden state for decoder must be zeros
        assert torch.all(hidden_states == 0.)

        for i in range(seq_length):
            # [nenv, 5] + [nenv, 2] = [nenv, 7]
            concat_state = torch.cat((state, z[i]), dim=-1) # nenv, 7
            concat_state = self.mlp_in(concat_state) # nenv, 64
            # rnn input: [1, nenv, 64], hidden: [1, nenv, 256]
            state, hidden_states = self.RNN(concat_state.unsqueeze(0), hidden_states)
            # reshape state back to [nenv, 256] -> [nenv, 5]
            state = self.output_linear(state.squeeze(0))
            outputs[i] = state

        return outputs, hidden_states

    # forward function for rl: returns critic output, actor features, and new rnn hidden state
    def forward(self, z, rnn_hxs, infer=False):
        if infer:
            # Test time
            seq_length = 1
            nenv = self.nenv # 12

        else:
            # Training time
            seq_length = self.seq_length # num_steps
            nenv = self.nenv // self.nminibatch # batch_size

        x, h_new = self._forward(z, rnn_hxs, seq_length, nenv)
        rnn_hxs['rnn'] = h_new.squeeze(0)  # [nenv, 12, 256]

        if infer:
            return x.squeeze(0), rnn_hxs
        else:
            # [seq_len, batch_size, 5] -> [batch_size, seq_len, 5]
            return x.permute(1, 0, 2), rnn_hxs


'''
MLP decoder network for trait prediction pretext task, 
only used in Morton et al

Reference: https://github.com/sisl/latent_driver/blob/master/train_bc/bc_policy.py
'''
class MLP(nn.Module):
    def __init__(self, obs_space_dict, config, task, latent_size):
        super(MLP, self).__init__()
        self.config = config
        self.is_recurrent = False
        self.task = task
        assert task in ['pretext_predict', 'rl_predict']

        self.human_num = obs_space_dict['spatial_edges'].shape[0]
        # number of edges per agent = total number of spatial edges / total number of nodes
        self.spatial_edge_num = int(self.human_num // obs_space_dict['pretext_nodes'].shape[0])


        self.seq_length = config.pretext.num_steps
        self.nenv = config.pretext.batch_size
        self.nminibatch = 1

        # init the parameters
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        # robot node size + robot temporal edge feature size + (one) robot spatial edge feature size
        # 1 + 2 + 1 = 4
        # self.robot_state_size = 1 + latent_size + obs_space_dict['pretext_temporal_edges'].shape[1]
        # input_size = self.robot_state_size + obs_space_dict['pretext_spatial_edges'].shape[1] * self.spatial_edge_num  # 6
        input_size = 2 + latent_size # 4

        self.mlp = nn.Sequential(
            init_(nn.Linear(input_size, 128)), nn.ReLU(),
            init_(nn.Linear(128, 128)), nn.ReLU(),
            init_(nn.Linear(128, 128)), nn.ReLU(),
            init_(nn.Linear(128, 1)), nn.Tanh())

        self.train()

    def _forward(self, inputs, z, seq_length, nenv):
        robot_node = inputs['pretext_nodes'].permute(1, 0, 2)  # seq_length, batch_size, 2
        spatial_edges = inputs['pretext_spatial_edges'].permute(1, 0, 2)  # # seq_length, batch_size, 2
        z = z.permute(1, 0, 2)  # seq_len, batch_size, 1

        # remove the action from robot_node and concat it with z
        # [seq_length, nenv, 1]+[seq_length, nenv, 1]+[seq_length, nenv, 2] = # [seq_length, nenv, 4]
        # concat_state: [px, delta_px w.r.t. front car, latent state z]
        concat_state = torch.cat((robot_node[:, :, 0, None], spatial_edges[:, :, 0, None], z), dim=-1)

        s = self.mlp(concat_state) # [seq_length, nenv, 1]
        return s

    # forward function for rl: returns critic output, actor features, and new rnn hidden state
    def forward(self, inputs, z, infer=False):
        if infer:
            # Test time
            seq_length = 1
            nenv = self.nenv # 12

        else:
            # Training time
            seq_length = self.seq_length # 30
            nenv = self.nenv // self.nminibatch # 12/2 = 6

        x = self._forward(inputs, z, seq_length, nenv)

        if self.task == 'rl':
            raise NotImplementedError("MLP_attn has not implemented as an RL policy yet")

        else:
            if infer:
                # get rid of seq_len dimension since seq_len = 1
                return x.squeeze(0)
            else:
                # [seq_length, batch_size, 1] -> [batch_size, seq_len]
                return x.permute(1, 0, 2).squeeze(-1)
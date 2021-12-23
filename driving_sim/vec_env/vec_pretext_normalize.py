import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import os

from torch.nn.utils.rnn import pad_sequence
from . import VecEnvWrapper

class VecPretextNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, config):
        VecEnvWrapper.__init__(self, venv)
        self.config = config
        self.device=torch.device("cuda:0" if self.config.training.cuda else "cpu")

        self.human_num = config.env_config.car.max_veh_num
        self.nenv = config.training.num_processes

        # for latent state inferrence model
        self.pred_model = None
        self.pred_model_hidden_states=None

        self.accuracy_list = [] # to save the prediction accuracy during training in csv

        self.step_counter = None
        # update prediction every num_steps steps
        self.pred_period = config.pretext.num_steps
        self.pred_inputs = None
        self.pred_outputs = None
        # key used for process traj
        self.input_key_list = ['pretext_nodes', 'pretext_spatial_edges', 'pretext_temporal_edges', 'pretext_masks', 'pretext_infer_masks', 'dones']
        # key used to forward pred_model
        self.save_key_list = ['pretext_nodes', 'pretext_spatial_edges', 'pretext_temporal_edges']


    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()

        # process the observations and reward
        obs = self.process_obs(obs, done)

        return obs, rews, done, infos


    def reset(self):
        obs = self.venv.reset()
        self.step_counter = 0
        # print('reset step counter!')
        self.agg_list_x, self.agg_list_y = [], []
        self.con_list_x, self.con_list_y = [], []
        # buffer to store 20 steps of observation for many2one pred_model
        # print(self.step_counter, 'init pred_inputs')
        self.pred_inputs = {}
        for key in self.input_key_list:
            self.pred_inputs[key] = []

        # initialize all predictions to be zeros
        self.pred_outputs = torch.zeros(self.nenv, self.human_num, 2, device=self.device)

        obs = self.process_obs(obs, np.zeros(self.num_envs))

        return obs

    # truncate and pad the pred_inputs
    # returns: mask for traj and their actual lengths
    def process_traj(self):
        # convert self.pred_inputs to torch tensors
        # print(self.step_counter, 'convert pred_inputs to tensor')
        pred_inputs_tensor = {}
        for key in self.input_key_list:
            pred_inputs_tensor[key] = torch.stack(self.pred_inputs[key], dim=0)

        # we only consider the cars which are not dummy AND in the correct lane
        humans_masks = torch.logical_and(pred_inputs_tensor['pretext_masks'], pred_inputs_tensor['pretext_infer_masks'])  # [traj_len, nenv, human_num]
        done_masks = pred_inputs_tensor['dones'] # [traj_len, nenv, 1]

        # add a sentinel in the front
        humans_masks = torch.cat((torch.zeros((1, self.nenv, self.human_num), dtype=torch.bool, device=self.device), humans_masks), dim=0)  # 21, nenv, human_num

        humans_start_idx = torch.logical_not(humans_masks).cumsum(dim=0).argmax(dim=0)

        done_start_idx = done_masks.cumsum(dim=0).argmax(dim=0)

        # [self.nenv, self.human_num]
        # 0 ~ 20
        start_idx = torch.maximum(humans_start_idx, done_start_idx.unsqueeze(-1))

        start_idx_flat = start_idx.view(self.nenv*self.human_num) # 0 ~ 20

        # remove the traj with zero length (start_idx = 20)
        # mask to indicate which traj are predicted & which cannot or does not need to be predicted
        if self.config.pretext.env_name == 'TIntersectionPredictFrontAct-v0':
            max_start_idx = self.pred_period - 2 # each traj has at least 2 steps
        else:
            max_start_idx = self.pred_period - 1
        mask = start_idx < max_start_idx

        seq_len = self.pred_period - start_idx_flat[start_idx_flat < max_start_idx]

        new_data = {'pretext_nodes': [],
                    'pretext_temporal_edges':[],
                    'pretext_spatial_edges': []}

        # slice the traj and save in return value
        for i in range(self.nenv):  # for each env
            for j in range(self.human_num):
                # if traj_len = 20, the largest max index is 18 (at least 2 steps)
                if start_idx[i, j] < max_start_idx:
                    for key in self.save_key_list:
                        # change the px of pretext_nodes to odometry (displacement since 20 steps ago)
                        if key == 'pretext_nodes':
                            if self.config.pretext.env_name == 'TIntersectionPredictFrontAct-v0':
                                pred_inputs_tensor[key][start_idx[i, j]:, i, j, 0] = pred_inputs_tensor[key][
                                                                                  start_idx[i, j]:, i, j, 0] \
                                                                                  - pred_inputs_tensor[key][
                                                                                      start_idx[i, j], i, j, 0]
                            else:
                                pred_inputs_tensor[key][start_idx[i, j]:, i, j] = pred_inputs_tensor[key][start_idx[i, j]:, i, j]\
                                                                                - pred_inputs_tensor[key][start_idx[i, j], i, j]
                        new_data[key].append(pred_inputs_tensor[key][start_idx[i, j]:, i, j])

        # deal with the 1 step offset between state and actions for Morton et al
        if self.config.pretext.env_name == 'TIntersectionPredictFrontAct-v0':
            # for each sequence in new_data
            for i in range(len(seq_len)):
                seq_len[i] = seq_len[i] - 1
                for key in self.save_key_list:
                    if key == 'pretext_nodes':
                        # take [s0, ..., s(t-1)] and [a1, ..., at]
                        new_data[key][i] = torch.stack((new_data[key][i][:-1, 0], new_data[key][i][1:, 1]), dim=-1)
                    # all keys should remove st
                    else:
                        new_data[key][i] = new_data[key][i][:-1, :]

        if len(seq_len) > 0:
            # pad the sequences
            for key in self.save_key_list:
                new_data[key] = pad_sequence(new_data[key])
            # error check
            if self.config.pretext.env_name == 'TIntersectionPredictFrontAct-v0':
                assert (new_data['pretext_nodes'][:, :, 0] >= 0).all()
            else:
                assert (new_data['pretext_nodes'] >= 0).all()

        return new_data, mask, seq_len


    # O: pretext_nodes: [nenv, human_num, 1], pretext_temporal_edges: [nenv, human_num, 1]
    # pretext_spatial_edges: [nenv, human_num, 2]
    def process_obs(self, O, done):
        self.step_counter = self.step_counter + 1
        # print(self.step_counter)

        # all pretext models takes inputs starting with "pretext"
        # also need pretext_masks and dones for calculating seq_len
        if self.config.pretext.env_name == 'TIntersectionPredictFrontAct-v0':
            for key in self.input_key_list:
                if key == 'dones':
                    self.pred_inputs[key].append(torch.from_numpy(done).bool().to(self.device))
                elif key == 'pretext_nodes':
                    self.pred_inputs[key].append(torch.cat((O[key], O['pretext_actions']), dim=-1))
                else:
                    self.pred_inputs[key].append(O[key])
        else:
            for key in self.input_key_list:
                if key == 'dones':
                    self.pred_inputs[key].append(torch.from_numpy(done).bool().to(self.device))
                else:
                    self.pred_inputs[key].append(O[key])


        # update belief of human intents every 20 steps
        # don't predict in the first 20 steps
        if len(self.pred_inputs['pretext_nodes']) == self.pred_period:
            model_inputs, mask, seq_len = self.process_traj()
            # if we didn't find any valid trajectories, don't run inference & keep the last pred_outputs
            if len(seq_len) > 0:
                # initialize the hidden states of pred_model to be zeros
                self.pred_model_hidden_states = {}
                self.pred_model_hidden_states['rnn'] = torch.zeros(len(seq_len),
                                                                   self.config.network.rnn_hidden_size,
                                                                   device=self.device)

                with torch.no_grad():
                    output = self.pred_model.predict(model_inputs, self.pred_model_hidden_states, seq_len.cpu())

                 # re-initialize the z of disappeared cars
                removed_car_mask = torch.logical_not(O['pretext_masks'])

                self.pred_outputs[removed_car_mask] = torch.zeros(*self.pred_outputs[removed_car_mask].size(), device=self.device)

                self.pred_outputs[mask] = output

                # normalize the z values
                self.pred_outputs = self.pred_outputs / 8.
                # self.pred_outputs = self.pred_outputs / 2.


                self.clear_pred_inputs()

                # for latent representation visualization only, commenting these out in RL training is recommended
                # plot the representations
                # agg = self.pred_outputs[torch.logical_and(O['true_labels'] == 0, mask)]
                # con = self.pred_outputs[torch.logical_and(O['true_labels'] == 1, mask)]
                # agg = agg.view(-1, 2).cpu().numpy()
                # con = con.view(-1, 2).cpu().numpy()
                # self.agg_list_x.extend(copy.deepcopy(agg[:, 0]))
                # self.agg_list_y.extend(copy.deepcopy(agg[:, 1]))
                # self.con_list_x.extend(copy.deepcopy(con[:, 0]))
                # self.con_list_y.extend(copy.deepcopy(con[:, 1]))


        # for latent representation visualization only, commenting these out in RL training is recommended
        # if self.step_counter % (self.pred_period*100) == 0 and self.step_counter > 0:
        #     self.fig_num = int(self.step_counter)
        #     plt.figure(self.fig_num)
        #
        #     plt.scatter(self.con_list_x, self.con_list_y, color='g', s=1)
        #     plt.scatter(self.agg_list_x, self.agg_list_y, color='r', s=1)
        #     plt.savefig(os.path.join(self.config.training.output_dir, str(self.fig_num) + '.png'), dpi=300)
        #     plt.close(self.fig_num)

        # replace the last digit of O['spatial_edges'] with pred
        O['spatial_edges'][:, :, -2:] = self.pred_outputs

        # the policy network only need the keys below (doesn't need keys starting with 'pretext')
        obs = {'robot_node': O['robot_node'],
                'spatial_edges': O['spatial_edges'],
                'temporal_edges': O['temporal_edges']}

        return obs

    def clear_pred_inputs(self):
        self.pred_inputs.clear()
        for key in self.input_key_list:
            self.pred_inputs[key] = []
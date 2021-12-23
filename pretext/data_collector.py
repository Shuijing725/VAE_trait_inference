import pickle
import copy
import os
import numpy as np

from rl.envs import make_vec_envs

'''
collect trajectory data of other cars in the env and the label of the cars' traits
make sure each trajectory has and only has one car

device: cpu or cuda0
train_data: True if collect training data, False if collect testing data
config: config object
'''
def collectMany2OneData(device, train_data, config):
    # always use 'TIntersectionPredictFrontAct-v0', since the observation is compatible for both our method and Morton baseline
    env_name = 'TIntersectionPredictFrontAct-v0'
    # for render
    env_num = 1 if config.pretext.render else config.pretext.num_processes

    human_num = config.env_config.car.max_veh_num

    # create parallel envs
    envs = make_vec_envs(env_name, config.env_config.env.seed, env_num,
                         config.env_config.reward.gamma, None, device, allow_early_resets=True, config=config,
                         wrap_pytorch=False)

    # key list for observation from env
    ob_key_list = ['pretext_nodes', 'pretext_spatial_edges', 'pretext_temporal_edges', 'labels', 'pretext_masks', 'dones']
    # key list for saved data
    save_key_list = ['pretext_nodes', 'pretext_spatial_edges', 'pretext_temporal_edges', 'labels']

    # collect data for pretext training
    # list of dicts, the value of each key is a list of 30
    data_list = [] # list for all data collected
    data = {} # buffer for data from env
    # initialize buffer to store data from env
    # each data[key] = list of traj_len, each element of the list = array (nenv, human_num, ?)
    for key in ob_key_list:
        data[key] = []

    obs = envs.reset()

    # 1 epoch -> 1 file
    for epoch in range(10):
        print('collect data epoch', epoch)
        # how may traj do we want in one file
        # number of collected traj in a file will be >= config.pretext.num_data_per_file
        while(len(data_list)) < config.pretext.num_data_per_file:

            if config.pretext.render:
                envs.render()

            # NOTE: the robot doesn't move!
            action = np.zeros((env_num, ), dtype=int)

            # save the previous obs before it is overwritten
            prev_obs = copy.deepcopy(obs)

            obs, rew, done, info = envs.step(action)

            # pretext node: [px, ax] of other cars
            pretext_nodes = np.concatenate((prev_obs['pretext_nodes'], prev_obs['pretext_actions']), axis=-1)
            data['pretext_nodes'].append(copy.deepcopy(pretext_nodes))
            data['pretext_spatial_edges'].append(copy.deepcopy(prev_obs['pretext_spatial_edges']))
            data['pretext_temporal_edges'].append(copy.deepcopy(prev_obs['pretext_temporal_edges']))
            data['labels'].append(copy.deepcopy(prev_obs['true_labels']))
            data['pretext_masks'].append(copy.deepcopy(prev_obs['pretext_masks']))
            data['dones'].append(copy.deepcopy(done))


            # save data to data_list for every 20 steps
            if len(data['labels']) == config.pretext.num_steps:
                # process traj, keep the last sub-traj of non-dummy human in each traj
                processed_data = process_traj(data, save_key_list, env_num, config.pretext.num_steps, human_num)
                data_list.extend(copy.deepcopy(processed_data))
                data.clear()
                for key in ob_key_list:
                    data[key] = []

        print('number of traj in a file:', len(data_list))
        # save observations as pickle files
        # observations is a list of dict [{'x':, 'intent':, 'u':}, ...]
        filePath = os.path.join(config.pretext.data_save_dir, 'train') if train_data \
            else os.path.join(config.pretext.data_save_dir, 'test')
        if not os.path.isdir(filePath):
            os.makedirs(filePath)
        filePath = os.path.join(filePath, str(epoch)+'.pickle')
        with open(filePath, 'wb') as f:
            pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        data_list.clear()

    envs.close()


'''
process the observation from env, and convert then to the saving format for training

input data: dictionary of nested lists
output return value: list of dictionaries of np array, where each dict = one traj

data: traj_len steps of observation data from env, each key has value with shape [traj_len, nenv, human_num, ?]
save_key_list: list of observation keys to be saved
nenv: number of parallel env
traj_len: max traj length of each traj, slice the input traj data into traj with length = traj_len
human_num: max number of other cars in env
'''
def process_traj(data, save_key_list, nenv, traj_len, human_num):
    new_data = []
    # convert each key in data to np array
    for key in data:
        data[key] = np.array(data[key])

    # calculate the start index for each traj
    humans_masks = np.array(data['pretext_masks']) # [traj_len, nenv, human_num]
    done_masks = np.expand_dims(np.array(data['dones']), axis=-1) # [traj_len, nenv, 1]

    # add a sentinel in the front
    humans_masks = np.concatenate((np.zeros((1, nenv, human_num)), humans_masks), axis=0) # 21, nenv, human_num
    humans_start_idx = np.logical_not(humans_masks).cumsum(axis=0).argmax(axis=0)
    done_masks = np.concatenate((np.zeros((1, nenv, 1), dtype=bool), done_masks), axis=0)
    done_start_idx = done_masks.cumsum(axis=0).argmax(axis=0)
    # if done_masks are all zeros, done_start_idx should be 0

    start_idx = np.maximum(humans_start_idx, done_start_idx)

    # slice the traj and save in return value
    for i in range(nenv):  # for each env
        for j in range(human_num):
            # if traj_len = 20, the largest max index is 18 (so that each traj has at least 2 steps)
            if start_idx[i, j] < traj_len-1:
            # the largest max index is 15 (so that each traj has at least 5 steps)
            # if start_idx[i, j] < traj_len - 4:
                cur_dict = {}
                for key in save_key_list:
                    # only save one label for each traj
                    if key == 'labels':
                        cur_dict[key] = data[key][-1, i, j]
                    else:
                        # data[key]: [traj_len, nenv, human_num, ?]
                        cur_dict[key] = data[key][start_idx[i, j]:, i, j]
                # change the px of pretext_nodes to odometry (displacement since 20 steps ago)
                cur_dict['pretext_nodes'][:, 0] = cur_dict['pretext_nodes'][:, 0] - cur_dict['pretext_nodes'][0, 0]
                # error check: all px must be non-negative
                assert (cur_dict['pretext_nodes'][:, 0] >= 0).all(), cur_dict['pretext_nodes'][:, 0]

                new_data.append(copy.deepcopy(cur_dict))
    return new_data

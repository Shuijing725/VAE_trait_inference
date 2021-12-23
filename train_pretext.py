from tqdm import trange
import os
import time
import pandas as pd
import shutil
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.backends.cudnn as cudnn


from rl import utils
from pretext.pretext_models.cvae_model import *

from pretext.loss import *
from pretext.data_loader import loadDataset

from configs.config import Config

from driving_sim.envs import *

# use the default config
config = Config

if __name__ == '__main__':
    device = torch.device("cuda" if config.training.cuda else "cpu")
    print("Using device:", device)

    # create a dummy env to get observation space
    envs = TIntersectionPredictFront()
    human_num = config.env_config.car.max_veh_num

    envs.configure(config.env_config)

    # initialize the network model and loss function
    # ours: rnn encoder + rnn decoder (cvae_decoder = 'lstm')
    # Morton et al: rnn encoder + mlp decoder (cvae_decoder = 'mlp')
    model=CVAEIntentPredictor(envs.observation_space.spaces, task='pretext_predict',
                              decoder_base=config.pretext.cvae_decoder,
                              config=config)
    loss_func = CVAE_loss(config=config, schedule_kl_method='constant')

    nn.DataParallel(model).to(device)

    # load data
    data_generator = loadDataset(train_data=True,
                                 batch_size=config.pretext.batch_size,
                                 num_workers=8,
                                 drop_last=True,
                                 load_dir=config.pretext.data_load_dir
                                 )

    print('load data complete')

    # save config files in the saving directory
    if not os.path.isdir(config.pretext.model_save_dir):
        os.makedirs(config.pretext.model_save_dir)
        shutil.copytree('configs', os.path.join(config.pretext.model_save_dir, 'configs'))

    if config.training.cuda:
        model.cuda().train()
    else:
        model.train()

    cudnn.benchmark = True

    # if resume from an existing model, load the model
    if config.pretext.resume_train:
        model.load_state_dict(torch.load(config.pretext.model_load_dir))

    # define the optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=config.pretext.lr,
                           weight_decay=1e-6)

    if not os.path.exists(os.path.join(config.pretext.model_save_dir, 'checkpoints')):
        os.makedirs(os.path.join(config.pretext.model_save_dir, 'checkpoints'))


    # the main training loop
    # for each epoch
    kl_weight = []
    for ep in trange(config.pretext.epoch_num, position=0):
        # print('training epoch', ep)
        # update the learning rate based on the lr schedule
        utils.update_linear_schedule(
            optimizer, ep, config.pretext.epoch_num,
            config.pretext.lr)
        # update the beta of cvae loss
        loss_func.schedule_kl_weight()

        loss_ep = []
        act_loss_ep = []
        kl_loss_ep = []

        # for debugging only
        # for _ in range(453):
        #     a = loss_func.forward()
        #     if _ == 0:
        #         kl_weight.append(a)


        # for each batch of data
        for n_iter, (robot_node, spatial_edges, temporal_edges, labels, seq_len) in enumerate(data_generator):
            # print(n_iter)
            robot_node = robot_node.float().to(device)
            spatial_edges = spatial_edges.float().to(device)
            temporal_edges = temporal_edges.float().to(device)
            labels = labels.float().to(device)

            # initialize rnn hidden state
            # encoder
            rnn_hxs_encoder = {}
            rnn_hxs_encoder['rnn'] = torch.zeros(config.pretext.batch_size,
                                         config.network.rnn_hidden_size,
                                         device=device)
            # decoder
            rnn_hxs_decoder = {}
            rnn_hxs_decoder['rnn'] = torch.zeros(config.pretext.batch_size,
                                                 config.network.rnn_hidden_size,
                                                 device=device)

            model.zero_grad()
            optimizer.zero_grad()

            # ours
            if config.pretext.cvae_decoder == 'lstm':
                # pretext nodes: [px], pretext_spatial_edges: relative [delta_px, delta_vx] with front car & ego car
                # pretext_temporal_edges: [vx]
                robot_node = robot_node[:, :, 0, None] # remove ax since the pretext nodes are [px, ax] in dataset
                state_dict = {'pretext_nodes': robot_node, 'pretext_spatial_edges': spatial_edges,
                              'pretext_temporal_edges': temporal_edges}
                # joint_states = torch.cat((robot_node, temporal_edges, spatial_edges), dim=-1)
                joint_states = torch.cat((robot_node, spatial_edges[:, :, 0, None]), dim=-1)
                pred_traj, z_mean, z_log_var, rnn_hxs_encoder, rnn_hxs_decoder, z = model(state_dict, rnn_hxs_encoder, rnn_hxs_decoder, seq_len)
                loss, act_loss, kl_loss = loss_func.forward(joint_states, pred_traj, z_mean, z_log_var, seq_len)

            # Morton et al
            else:
                # pretext nodes: [px, action], pretext_spatial_edges: relative [delta_px, delta_vx] with front car & ego car
                # pretext_temporal_edges: [vx]
                state_dict = {'pretext_nodes': robot_node, 'pretext_spatial_edges': spatial_edges,
                              'pretext_temporal_edges': temporal_edges}
                pred_act, z_mean, z_log_var, rnn_hxs_encoder, rnn_hxs_decoder, z = model(state_dict, rnn_hxs_encoder, rnn_hxs_decoder, seq_len)
                loss, act_loss, kl_loss = loss_func.forward(robot_node[:, :, 1], pred_act, z_mean, z_log_var, seq_len)


            loss.backward()
            optimizer.step()

            loss_ep.append(loss.item())
            act_loss_ep.append(act_loss.item())
            kl_loss_ep.append(kl_loss.item())

        # save checkpoints every 5 epochs
        if ep % 5 == 0:
            fname = os.path.join(config.pretext.model_save_dir, 'checkpoints', str(ep)+'.pt')
            torch.save(model.state_dict(), fname)
            print('Model saved to '+fname)
        if ep % config.pretext.log_interval == 0:
            end = time.time()

            # log the beta for cvae
            print("Epoch {}, loss {}, action loss {}, kl loss {}, beta {}\n"
                  .format(ep, round(sum(loss_ep), 3), round(sum(act_loss_ep), 3), round(sum(kl_loss_ep), 3), loss_func.beta))
            df = pd.DataFrame({'epoch': [ep], 'loss': [sum(loss_ep)], 'act_loss': [sum(act_loss_ep)],
                               'kl_loss': [sum(kl_loss_ep)], 'beta':[loss_func.beta]})


            if os.path.exists(os.path.join(config.pretext.model_save_dir, 'progress.csv')) and ep > config.pretext.log_interval:
                df.to_csv(os.path.join(config.pretext.model_save_dir, 'progress.csv'), mode='a', header=False, index=False)
            else:
                df.to_csv(os.path.join(config.pretext.model_save_dir, 'progress.csv'), mode='w', header=True, index=False)

    envs.close()
    print('Pretext Training Complete')
    # for debugging only
    # epochs = np.arange(config.pretext.epoch_num)
    # plt.plot(epochs, kl_weight)
    # plt.show()

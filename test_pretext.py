import matplotlib.pyplot as plt
import argparse
import logging
import sys
import torch.nn as nn

from pretext.pretext_models.cvae_model import CVAEIntentPredictor

from pretext.data_loader import *
from pretext.loss import *

from driving_sim.envs import *


if __name__ == '__main__':
    # the following parameters will be determined for each test run
    parser = argparse.ArgumentParser('Parse configuration file')
    # the model directory that we are testing
    parser.add_argument('--model_dir', type=str, default='data_sl/models/public_ours')
    # the directory of testing dataset
    parser.add_argument('--data_load_dir', type=str, default='data_sl/data/public_dataset')
    # whether plot a batch of original traj and reconstructed traj and save the figures
    # only works for our method since Morton and Kochenderfer baseline does not reconstructed the traj
    parser.add_argument('--save_plot', default=False, action='store_true')
    # model weight file you want to test
    parser.add_argument('--test_model', type=str, default='995.pt')
    test_args = parser.parse_args()

    # create folder for saving plots
    save_path = os.path.join(test_args.model_dir, test_args.test_model[:-3])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # get algorithm arguments and env configs
    from importlib import import_module

    model_dir_temp = test_args.model_dir
    if model_dir_temp.endswith('/'):
        model_dir_temp = model_dir_temp[:-1]

    # import config class from saved directory
    # if not found, import from the default directory
    try:
        model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
        model_arguments = import_module(model_dir_string)
        Config = getattr(model_arguments, 'Config')
    except:
        print('Failed to get Config function from ', test_args.model_dir)
        from configs.config import Config
    config = Config()


    device = torch.device("cuda" if config.training.cuda else "cpu")
    print("Using device:", device)

    # create a dummy env to get observation space
    envs = TIntersectionPredictFront()
    human_num = config.env_config.car.max_veh_num

    envs.configure(config.env_config)

    # initialize the NN model and loss function
    task = 'pretext_predict'
    # full cvae
    # ours: rnn encoder + rnn decoder (cvae_decoder = 'lstm')
    # Morton et al: rnn encoder + mlp decoder (cvae_decoder = 'mlp')
    model = CVAEIntentPredictor(envs.observation_space.spaces, task='pretext_predict',
                                decoder_base=config.pretext.cvae_decoder,
                                config=config)
    loss_func = CVAE_loss(config=config, schedule_kl_method='constant')


    nn.DataParallel(model).to(device)

    if config.training.cuda:
        model.cuda().eval()
    else:
        model.eval()

    # load the weights of the model
    model.load_state_dict(torch.load(os.path.join(test_args.model_dir, 'checkpoints', test_args.test_model), map_location=device))
    print('model load complete')


    # init log
    log_file = os.path.join(test_args.model_dir, 'test')
    if not os.path.exists(log_file):
        os.mkdir(log_file)
    log_file = os.path.join(test_args.model_dir, 'test', 'test_' + test_args.test_model + '.log')

    file_handler = logging.FileHandler(log_file, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # load testing data
    data_generator = loadDataset(train_data=False,
                                 batch_size=config.pretext.batch_size,
                                 num_workers=8,
                                 drop_last=True,
                                 load_dir=test_args.data_load_dir)
    print('load data complete')


    tot_loss=[]
    tot_act_loss = []
    tot_kl_loss = []
    aggressive_x_list, aggressive_y_list = [], []
    conservative_x_list, conservative_y_list = [], []
    zeros_x_list, zeros_y_list = [], []

    avg_accuracy = []
    text_list = [] # for latent space visualization
    num_outliners = 0
    # for each batch of data
    for n_iter, (robot_node, spatial_edges, temporal_edges, labels, seq_len) in enumerate(data_generator):
        robot_node = robot_node.float().to(device)
        spatial_edges = spatial_edges.float().to(device)
        temporal_edges = temporal_edges.float().to(device)
        labels_torch = labels.float().to(device)
        masks = torch.ones(config.pretext.batch_size, config.pretext.num_steps, 1).float().to(device)  # batch_size, seq_len, 1

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

        # forward model
        # Morton et al
        if config.pretext.cvae_decoder == 'mlp':
            state_dict = {'pretext_nodes': robot_node, 'pretext_spatial_edges': spatial_edges,
                          'pretext_temporal_edges': temporal_edges}
            with torch.no_grad():
                pred_act, z_mean, z_log_var, rnn_hxs_encoder, rnn_hxs_decoder, z = model(state_dict,
                                                                                     rnn_hxs_encoder,
                                                                                     rnn_hxs_decoder, seq_len)
            loss, act_loss, kl_loss = loss_func.forward(robot_node[:, :, 1], pred_act, z_mean, z_log_var,
                                                        seq_len)
        # ours
        else:
            robot_node = robot_node[:, :, 0, None]
            state_dict = {'pretext_nodes': robot_node, 'pretext_spatial_edges': spatial_edges,
                          'pretext_temporal_edges': temporal_edges}
            # joint_states = torch.cat((robot_node, temporal_edges, spatial_edges), dim=-1)
            joint_states = torch.cat((robot_node, spatial_edges[:, :, 0, None]), dim=-1)
            with torch.no_grad():
                pred_traj, z_mean, z_log_var, rnn_hxs_encoder, rnn_hxs_decoder, z = model(state_dict,
                                                                                          rnn_hxs_encoder,
                                                                                          rnn_hxs_decoder,
                                                                                          seq_len)
            loss, act_loss, kl_loss = loss_func.forward(joint_states, pred_traj, z_mean, z_log_var, seq_len, train=False)


        tot_loss.append(loss.cpu().numpy())
        tot_act_loss.append(act_loss.cpu().numpy())
        tot_kl_loss.append(kl_loss.cpu().numpy())
        z = z.to('cpu').numpy()

        # save the inferred representation for each traj for visualization
        aggressive_x = z[:, 0][labels == 0.]
        conservative_x = z[:, 0][labels == 1.]
        aggressive_y = z[:, 1][labels == 0.]
        conservative_y = z[:, 1][labels == 1.]

        aggressive_x_list.extend(list(aggressive_x))
        aggressive_y_list.extend(list(aggressive_y))
        conservative_x_list.extend(list(conservative_x))
        conservative_y_list.extend(list(conservative_y))


        # save the traj, labels, and inferred z for one batch for visualization
        if test_args.save_plot and n_iter == 7:
            target_pos = robot_node.squeeze(-2).to("cpu").numpy() # [batch_size * human_num, 50, 3]
            front_ego_pos = spatial_edges.squeeze(-2).to("cpu").numpy()

            labels = labels.to("cpu").numpy()

            for i in range(len(labels)):
                if i % 10 == 0: # or z[i, 0] < -5:
                    # plot the reconstructed traj
                    if config.pretext.cvae_decoder == 'lstm':
                        plt.figure(i)
                        pred_target_pos = pred_traj[i, :seq_len[i], 0].to("cpu").numpy()
                        # pred_front_ego_pos = pred_traj[i, :seq_len[i], 2].to("cpu").numpy()
                        pred_front_ego_pos = pred_traj[i, :seq_len[i], 1].to("cpu").numpy()
                        # plot the target car
                        plt.scatter(target_pos[i, :seq_len[i], 0], np.zeros((seq_len[i], )), s=5, marker='o',
                                    c=np.arange(seq_len[i]), cmap='autumn_r')
                        # plot the front car
                        plt.scatter(front_ego_pos[i,:seq_len[i],0]+target_pos[i,:seq_len[i],0],np.zeros((seq_len[i], )),
                                    s=5,marker='<', c=np.arange(seq_len[i]), cmap='spring_r')
                        # plot the reconstructed target car
                        plt.scatter(pred_target_pos, np.ones((seq_len[i],)), s=5, marker='o',
                                    c=np.arange(seq_len[i]), cmap='summer_r')
                        # plot the reconstructed front car
                        plt.scatter(pred_front_ego_pos +pred_target_pos, np.ones((seq_len[i],)),
                                    s=5, marker='<', c=np.arange(seq_len[i]), cmap='winter_r')

                        # add text for z and true label to this figure
                        x_lower, x_upper = plt.xlim()
                        y_lower, y_upper = plt.ylim()
                        plt.text(0.2, 0.6, 'length:'+str(seq_len[i]))
                        # plt.text(2, 0, 'true traj: below')
                        plt.text(x_lower+0.2*(x_upper-x_lower), y_lower + 0.2*(y_upper-y_lower), 'z: ('+str(round(z[i, 0], 2))+', ' +str(round(z[i, 1], 2))+')')
                        plt.text(x_lower+0.2*(x_upper-x_lower), y_lower + 0.4*(y_upper-y_lower), 'true label: '+str(labels[i]))

                        plt.savefig(os.path.join(save_path, str(i)+'.png'), dpi=300)
                        plt.close(i)
                    # add figure number (i) to the latent state plot: z_x, z_y, i
                    text_list.append((z[i, 0], z[i, 1], i))


    logging.info('average accuracy: {:.4f}'.format(np.mean(avg_accuracy)))
    logging.info('average loss: %.4f, action loss: %.4f, kl loss: %.4f', np.mean(tot_loss),
                 np.mean(tot_act_loss), np.mean(tot_kl_loss))

    # plot the distribution of z
    plot_portion = 1
    agg_last_idx = int(len(aggressive_x_list) * plot_portion)
    con_last_idx = int(len(conservative_x_list) * plot_portion)
    plt.figure(0)

    # interwave two classes of points to better visualize the overlaps
    seg_num = 25
    for i in range(seg_num):
        start_agg = len(aggressive_x_list) // seg_num * i
        start_con = len(conservative_x_list) // seg_num * i
        if i == seg_num - 1:
            end_agg, end_con = -1, -1
            plt.scatter(aggressive_x_list[start_agg:end_agg], aggressive_y_list[start_agg:end_agg],
                        color='orangered',
                        s=0.5, label='Aggressive')
            plt.scatter(conservative_x_list[start_con:end_con], conservative_y_list[start_con:end_con],
                        color='royalblue',
                        s=0.5,
                        label='Conservative')
        else:
            end_agg = len(aggressive_x_list) // seg_num * (i + i)
            end_con = len(conservative_x_list) // seg_num * (i + 1)
            plt.scatter(aggressive_x_list[start_agg:end_agg], aggressive_y_list[start_agg:end_agg],
                        color='orangered',
                        s=0.5)
            plt.scatter(conservative_x_list[start_con:end_con], conservative_y_list[start_con:end_con],
                        color='royalblue',
                        s=0.5)

    lgnd = plt.legend(loc=2, fontsize=13)
    # change the marker size manually for both lines
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]


    for i in range(len(text_list)):
        plt.text(text_list[i][0], text_list[i][1], text_list[i][2], fontsize=4)

    plt.title('Ours', fontsize=20)
    plt.savefig(os.path.join(save_path, 'latent_space.png'), dpi=300)
    plt.close(0)

    plt.figure(1)
    plt.scatter(aggressive_x_list[:agg_last_idx], aggressive_y_list[:agg_last_idx], color='r', s=5, label='Aggressive')
    plt.savefig(os.path.join(save_path, 'latent_space_agg.png'), dpi=300)
    plt.close(1)
    plt.figure(2)
    plt.scatter(conservative_x_list[:con_last_idx], conservative_y_list[:con_last_idx], color='g', s=5, label='Conservative')
    plt.savefig(os.path.join(save_path, 'latent_space_con.png'), dpi=300)
    plt.close(2)
    # plt.show()





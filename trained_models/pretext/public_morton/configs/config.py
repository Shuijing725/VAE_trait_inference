import torch
from .driving_config import DrivingConfig


class BaseConfig(object):
    def __init__(self):
        pass

'''
Configuration file for PPO training, pretext trait inference training, pretext data collection, and network architectures for pretext and RL
'''
class Config(object):
    # initialize config file for the driving env
    env_config = DrivingConfig

    # ppo config
    ppo = BaseConfig()
    ppo.num_mini_batch = 2  # number of batches for ppo
    ppo.num_steps = 30  # number of forward steps
    ppo.recurrent_policy = True  # use a recurrent policy
    ppo.epoch = 5  # number of ppo epochs
    ppo.clip_param = 0.2  # ppo clip parameter
    ppo.value_loss_coef = 0.5  # value loss coefficient
    ppo.entropy_coef = 0.01 # entropy term coefficient
    ppo.use_gae = True  # use generalized advantage estimation
    ppo.gae_lambda = 0.95  # gae lambda parameter


    # training config for RL
    training = BaseConfig()
    training.render = False # render in training or not
    training.lr = 1e-4 # learning rate
    training.eps = 1e-5  # RMSprop optimizer epsilon
    training.alpha = 0.99  # RMSprop optimizer alpha
    training.max_grad_norm = 0.5  # max norm of gradients
    training.num_env_steps = 10e6  # number of environment steps to train: 10e6 for holonomic, 20e6 for unicycle
    training.use_linear_lr_decay = True  # use a linear schedule on the learning rate: True for unicycle, False for holonomic
    training.save_interval = 200  # save interval, one save per n updates
    training.log_interval = 20  # log interval, one log per n updates
    training.use_proper_time_limits = False  # compute returns taking into account time limits
    training.cuda_deterministic = False  # sets flags for determinism when using CUDA (potentially slow!)
    training.no_cuda = False  # disables CUDA training
    training.cuda = not training.no_cuda and torch.cuda.is_available()
    training.num_processes = 12  # how many training CPU processes to use
    training.output_dir = 'data/dummy'  # the saving directory for train.py
    training.resume = False  # resume training from an existing checkpoint or not
    training.load_path = None  # if resume = True, load from the following checkpoint
    training.overwrite = True  # whether to overwrite the output directory in training
    training.num_threads = 1  # number of threads used for intraop parallelism on CPU
    # if we use intent predictor in training, the path of the predictor model
    # must match the pretext.method and pretext.cvae_decoder below!
    training.pretext_model_path = 'data_sl/models/many2one/lstm_mlp_cvae_noVel/checkpoints/995.pt'


    # pretext task/trait inference config
    pretext = BaseConfig()
    # determine the size of representation space -> ob space for env
    env_config.ob_space.latent_size = 2
    # lstm (ours) or mlp (Morton et al)
    pretext.cvae_decoder = 'mlp'
    # we are running our method
    if pretext.cvae_decoder == 'lstm':
        pretext.env_name = 'TIntersectionPredictFront-v0'
    # we are running Morton et al
    elif pretext.cvae_decoder == 'mlp':
        # ours: 'TIntersectionPredictFront-v0', Morton baseline: 'TIntersectionPredictFrontAct-v0'
        pretext.env_name ='TIntersectionPredictFrontAct-v0'
    else:
        raise ValueError("unknown pretext VAE decoder")
    pretext.num_processes = 16 # how many training CPU processes to use
    pretext.resume_train = False # whether we resume training from a previous checkpoint
    pretext.model_load_dir = 'data_sl/models/model_1human_all_policies/checkpoints/495.pt'  # path of the checkpoint we resume from

    # for data collection
    # how many traj to collect for each data file
    pretext.num_data_per_file = 4600
    # the directory that the dataset is saved in
    pretext.data_save_dir = 'data_sl/data/public_dataset'
    pretext.num_steps = 20  # number of steps per trajectory

    # for VAE training
    # where are we loading the data from
    pretext.data_load_dir = 'data_sl/data/public_dataset'
    pretext.lr = 5e-4 # pretext learning rate
    pretext.epoch_num = 1000 # number of epochs for pretext task
    pretext.batch_size = 1024 # batch size during training
    pretext.render = False # whether we render the pretext env in training or testing
    pretext.model_save_dir = 'data_sl/models/public_morton' # the directory for saving sl model
    pretext.log_interval = 1 # how often do we log the training info


    # network architecture config
    network = BaseConfig()
    network.rnn_hidden_size = 256  # Size of Human Human Edge RNN hidden state

    # each human car's state: observable state + inferred trait
    network.human_state_input_size = 2 + env_config.ob_space.latent_size

    # Embedding size
    network.embedding_size = 64  # Embedding size of edge features



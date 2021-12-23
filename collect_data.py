import torch

from configs.config import Config

from pretext.data_collector import *

if __name__ == '__main__':
    config = Config()
    device = torch.device("cuda" if config.training.cuda else "cpu")
    print("Using device:", device)

    save_dir_parent = config.pretext.data_save_dir
    print ("data saving directory:", save_dir_parent)
    # a list of [pretext.collect_train_data, pretext.data_save_dir]
    arg_list = [
                (True, os.path.join(save_dir_parent, 'train1')),
                (True, os.path.join(save_dir_parent, 'train2')),
                (True, os.path.join(save_dir_parent, 'train3')),
                (True, os.path.join(save_dir_parent, 'train4')),
                (True, os.path.join(save_dir_parent, 'train5')),
                (True, os.path.join(save_dir_parent, 'train6')),
                (True, os.path.join(save_dir_parent, 'train7')),
                (True, os.path.join(save_dir_parent, 'train8')),
                (True, os.path.join(save_dir_parent, 'train9')),
                (True, os.path.join(save_dir_parent, 'train10')),
                (False, os.path.join(save_dir_parent, 'test1')),
                (False, os.path.join(save_dir_parent, 'test2')),
                (False, os.path.join(save_dir_parent, 'test3')),
                (False, os.path.join(save_dir_parent, 'test4')),
                (False, os.path.join(save_dir_parent, 'test5')),
                ]

    for train_data, save_dir in arg_list:
        # change config object
        config.pretext.collect_train_data = train_data
        config.pretext.data_save_dir = save_dir

        # randomize the seed each run (to avoid manual change of seed)
        config.env_config.env.seed = np.random.randint(0, np.iinfo(np.uint32).max)
        print("Using seed", config.env_config.env.seed)

        collectMany2OneData(device, config.pretext.collect_train_data, config)

        print('Data Collection Complete', end='\n')
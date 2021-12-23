import pickle
import torch
import os
import glob

from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence


class SLDataset(Dataset):
    def __init__(self, picklePath):
        self.filePath=picklePath
        with open(self.filePath, 'rb') as f:
            self.ground_truth_data = pickle.load(f)
        return

    def __getitem__(self, index):
        return torch.tensor(self.ground_truth_data[index]['pretext_nodes']), \
               torch.tensor(self.ground_truth_data[index]['pretext_spatial_edges']), \
               torch.tensor(self.ground_truth_data[index]['pretext_temporal_edges']), \
               torch.tensor(self.ground_truth_data[index]['labels'])

    def __len__(self):
        return len(self.ground_truth_data)

'''
custom data loader function to deal with trajectories with changing length
'''
def pad_collate(batch):
  nodes, spatial_edges, temporal_edges, labels = zip(*batch)
  lengths = [len(node) for node in nodes]

  nodes_pad = pad_sequence(nodes, batch_first=True)
  spatial_edges_pad = pad_sequence(spatial_edges, batch_first=True)
  temporal_edges_pad = pad_sequence(temporal_edges, batch_first=True)

  return nodes_pad, spatial_edges_pad, temporal_edges_pad, torch.tensor(labels), lengths

'''
Load the collected dataset from pickle files to torch gpu, returns the data generator for the training loop

train_data = True: load data from algo_args.sl_data_load_dir/train; else load from algo_args.sl_data_load_dir/test
batch_size, num_workers, drio_last: arguments for torch dataloader
load_dir: the directory of the dataset to be loaded
'''

def loadDataset(train_data, batch_size, num_workers, drop_last, load_dir):
    assert os.path.exists(load_dir)
    print('Load data from', load_dir)
    all_datasets = []

    folder_name = 'train' if train_data else 'test'
    all_file_paths = glob.glob(os.path.join(load_dir, '*',folder_name,'*.pickle'))

    for j, filePath in enumerate(all_file_paths):
        all_datasets.append(SLDataset(picklePath=filePath))

    final_dataset = ConcatDataset(all_datasets)
    print('total number of data:', len(final_dataset))
    # for training: it is good to shuffle the dataset
    # for testing: not shuffle to trace some specific test cases easily
    shuffle = True if train_data else False

    # define the data generator for the training loop in the main function
    generator = torch.utils.data.DataLoader(final_dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            drop_last=drop_last,
                                            collate_fn=pad_collate)

    return generator


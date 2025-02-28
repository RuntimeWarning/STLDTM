import os
import re
import cv2
import h5py
import torch
import random
import datetime
import numpy as np
import pandas as pd
from torch.utils import data
from torchvision import transforms
from core.helpers.dataset_sevir import SEVIRTorchDataset


class Shanghai_Datasets(data.Dataset):
    def __init__(self, path, train=True):
        self.path = path
        if train:
            self.csvdata = pd.read_csv(path + '/Shanghai_train.csv', header=None)
        else:
            self.csvdata = pd.read_csv(path + '/Shanghai_test.csv', header=None)
        self.length = len(self.csvdata)
        self.tf = transforms.Resize((128, 128))

    def __getitem__(self, index):
        # img_start = np.random.randint(0, 26) #40-15+1=26
        sequence = list(self.csvdata.iloc[index])#[img_start:img_start+15]
        t = []
        for j in range(len(sequence)):
            img = cv2.imread(os.path.join(self.path, sequence[j]),
                             cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img = torch.from_numpy(img/255.0)
            t.append(img)
        data = torch.stack(t, dim=0)
        data = self.tf(data)
        return data

    def __len__(self):
        return self.length


class FeatureDataset(data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def __len__(self):
        length = len(os.listdir(self.path))
        return length

    def __getitem__(self, idx):
        path = os.path.join(self.path, f'{idx}.npy')
        z = np.load(path, allow_pickle=True)
        return torch.from_numpy(z)



class CIKM_Datasets(data.Dataset):
    def __init__(self, path, mode='train'):
        self.dataset = h5py.File(path, 'r', rdcc_nbytes=512**3)[mode]
        self.size = self.dataset.shape[0]
        self.transform = transforms.CenterCrop((128, 128)) # transform 101x101 to 128x128

    def __getitem__(self, index):
        data = self.dataset[index] / 255.0
        data = torch.from_numpy(data).type(torch.float32)
        data = self.transform(data)
        return data

    def __len__(self):
        return self.size
    

def get_datasets(name='cikm', opt='train', batch_size=16, num_workers=4, shuffle=True, **kwargs):
    if name == 'cikm':
        if opt == 'train':
            train_dataset = CIKM_Datasets(path='../PN_Datasets/CIKM2017.h5', mode='train')
            train_input_handle = data.DataLoader(train_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=shuffle)
            return train_input_handle
        elif opt == 'test':
            test_dataset = CIKM_Datasets(path='../PN_Datasets/CIKM2017.h5', mode='test')
            test_input_handle = data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=shuffle)
            return test_input_handle
        else:
            val_dataset = CIKM_Datasets(path='../PN_Datasets/CIKM2017.h5', mode='validation')
            val_input_handle = data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=shuffle)
            return val_input_handle
    elif name == 'shanghai':
        if opt == 'train':
            train_dataset = Shanghai_Datasets(path='../PN_Datasets/Sh_Radar_Data', train=True)
            train_input_handle = data.DataLoader(train_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=shuffle)
            return train_input_handle
        else:
            test_dataset = Shanghai_Datasets(path='../PN_Datasets/Sh_Radar_Data', train=False)
            test_input_handle = data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=shuffle)
            return test_input_handle
    elif 'feature' in name:
        dataset_name = name.split('_')[0]
        if opt == 'train':
            train_dataset = FeatureDataset(path=f'../PN_Datasets/{dataset_name}_128_features/train')
            train_input_handle = data.DataLoader(train_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=shuffle)
            return train_input_handle
        else:
            test_dataset = FeatureDataset(path=f'../PN_Datasets/{dataset_name}_128_features/test')
            test_input_handle = data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=shuffle)
            return test_input_handle
    elif name == 'sevir':
        train_valid_split = (2019, 1, 1)
        valid_test_split = (2019, 6, 1)
        if opt == 'train':
            train = SEVIRTorchDataset(
                dataset_dir='../PN_Datasets/SEVIR',
                split_mode='uneven',
                img_size=128,
                shuffle=shuffle,
                seq_len=25,
                stride=5,      # ?
                sample_mode='sequent',
                batch_size=batch_size,
                num_shard=1,
                rank=0,
                start_date=None,
                end_date=datetime.datetime(*train_valid_split),
                output_type=np.float32,
                preprocess=True,
                rescale_method='01',
                verbose=False
            )
            return train.get_torch_dataloader(num_workers=num_workers)
        elif opt == 'validation':
            val = SEVIRTorchDataset(
                dataset_dir='../PN_Datasets/SEVIR',
                split_mode='uneven',
                img_size=128,
                shuffle=shuffle,
                seq_len=25,
                stride=5,      # ?
                sample_mode='sequent',
                batch_size=batch_size,
                num_shard=1,
                rank=0,
                start_date=datetime.datetime(*train_valid_split),
                end_date=datetime.datetime(*valid_test_split),
                output_type=np.float32,
                preprocess=True,
                rescale_method='01',
                verbose=False
            )
            return val.get_torch_dataloader(num_workers=num_workers)
        else:
            test = SEVIRTorchDataset(
                dataset_dir='../PN_Datasets/SEVIR',
                split_mode='uneven',
                shuffle=shuffle,
                img_size=128,
                seq_len=25,
                stride=5,      # ?
                sample_mode='sequent',
                batch_size=batch_size,
                num_shard=1,
                rank=0,
                start_date=datetime.datetime(*valid_test_split),
                end_date=None,
                output_type=np.float32,
                preprocess=True,
                rescale_method='01',
                verbose=False
            )
            return test.get_torch_dataloader(num_workers=num_workers)
    else:
        raise ValueError('Unknown dataset name: {}'.format(name))
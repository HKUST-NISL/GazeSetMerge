import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import cv2
from PIL import Image
from torchvision import transforms
from datasets.one_data import OneData
from datasets.data_transforms import create_transform

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.span = len(datasets[1]) // len(datasets[0])

    def __getitem__(self, i):
        # k = np.random.randint(len(self.datasets[-1]) - i) + i
        seed_i = np.random.randint(len(self.datasets[-1]))
        np.random.seed(i + seed_i)
        k = np.random.randint(len(self.datasets[-1]))
        return tuple([self.datasets[0][i], self.datasets[1][k]])

    def __len__(self):
        return min(len(d) for d in self.datasets)


def create_combined_loader(data_dir, input_size, batch_size, data_type='mpii', test_ids=[]):
    transform = create_transform(input_size, is_train=True)
    eth_data = OneData(data_dir=data_dir, input_size=input_size, data_type='eth', transform=transform)
    add_data = OneData(data_dir=data_dir, input_size=input_size, data_type=data_type, 
        test_ids=test_ids, transform=transform)

    cmb_data = ConcatDataset(add_data, eth_data)

    loader = torch.utils.data.DataLoader(
        cmb_data,
        batch_size=batch_size//2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    return loader


def combine_zip_data(data):
    data_cmb = {}
    data_cmb['image'] = torch.cat([data[0]['image'], data[1]['image']])
    data_cmb['gaze'] = torch.cat([data[0]['gaze'], data[1]['gaze']])
    data_cmb['pose'] = torch.cat([data[0]['pose'], data[1]['pose']])
    data_cmb['left_eye_box'] = torch.cat([data[0]['left_eye_box'], data[1]['left_eye_box']])
    data_cmb['right_eye_box'] = torch.cat([data[0]['right_eye_box'], data[1]['right_eye_box']])
    data_cmb['type'] = torch.cat([data[0]['type'], data[1]['type']])

    return data_cmb


if __name__ == '__main__':

    loader = create_combined_loader('./data/GazeSetsPng', (224, 224), 64, 'mpii', [0])
    print(len(loader))
    for epoch in range(5):

        for i, data in enumerate(loader):
            # if i % 10000 != 0:
            #     continue

            # data_cmb = combine_zip_data(data)
            # print(i, data_cmb['image'].shape, data_cmb['gaze'].shape)

            if i == 0:
                print(i, data[0]['gaze'][:5], data[1]['gaze'][:5])
            
            break


import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2


class XGazeSubDataset(Dataset):
    def __init__(self, data_dir, data_name, transform=None):
        self.sub_id = os.path.splitext(os.path.basename(data_name))[0]
        self.data_path = os.path.join(data_dir, data_name)
        self.transform = transform

        with h5py.File(self.data_path, 'r') as data_fid:
            self.data_len = data_fid["face_patch"].shape[0]

    def __getitem__(self, index):
        
        with h5py.File(self.data_path, 'r') as data_fid:
            face = data_fid['face_patch'][index, :]
            pose = data_fid['face_head_pose'][index, :]

            if 'face_gaze' in data_fid.keys():
                gaze = data_fid['face_gaze'][index, :]
                x = {'sub_id':self.sub_id, 'face': face, 'gaze': gaze, 'pose': pose}
            else:    
                x = {'sub_id':self.sub_id, 'face': face, 'pose': pose}

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self):

        return self.data_len


def xgaze_dataset(data_dir, transform=None, is_train=True, is_test=False):

    dataset_dir = data_dir

    assert os.path.exists(dataset_dir)

    if is_train:
        lst_path = os.path.join(data_dir, 'train_lst.txt')
        with open(lst_path) as f:
            data_names = f.read().splitlines()
    
        train_dataset = torch.utils.data.ConcatDataset([
            XGazeSubDataset(data_dir, data_name, transform)
            for data_name in data_names
        ])
        return train_dataset

    else:
        if is_test:
            lst_path = os.path.join(data_dir, 'test_lst.txt')
        else:
            lst_path = os.path.join(data_dir, 'eval_lst.txt')
        with open(lst_path) as f:
            data_names = f.read().splitlines()
    
        test_dataset = torch.utils.data.ConcatDataset([
            XGazeSubDataset(data_dir, data_name, transform)
            for data_name in data_names
        ])
        return test_dataset


if __name__ == '__main__':

    sub_data = xgaze_dataset(data_dir='./data/xgaze_224/', is_train=True)

    print(len(sub_data))

    for i, data in enumerate(sub_data):

        print(i, data['face'].shape, data['gaze'].shape, data['pose'].shape)
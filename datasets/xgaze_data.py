import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List

from datasets.data_transforms import create_transform


def get_train_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True,
                           is_load_pose=False,
                           use_tiny=False,
                           input_size=224, 
                           is_train=True):
    # load dataset
    if use_tiny:
        refer_list_file = os.path.join(data_dir, 'train_test_split_small.json')
    else:
        refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'
    trans_train = create_transform(input_size, is_train)
    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=trans_train, is_shuffle=is_shuffle, is_load_label=True, is_load_pose=is_load_pose)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader


def get_test_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=False,
                           is_load_label=False,
                           is_load_pose=False,
                           use_tiny=False,
                           input_size=224):
    # load dataset
    if use_tiny:
        refer_list_file = os.path.join(data_dir, 'train_test_split_small.json')
    else:
        refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'test'
    trans = create_transform(input_size, False)
    test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                           transform=trans, is_shuffle=is_shuffle, is_load_label=is_load_label, is_load_pose=is_load_pose)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader


class GazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
                 index_file=None, is_load_label=True, is_load_pose=False):
        self.path = dataset_path
        self.hdfs = {}
        self.hdfs_lmk = {}
        # self.sub_folder = sub_folder
        self.sub_folder = 'all'
        self.is_load_label = is_load_label
        self.is_load_pose = is_load_pose
        self.sub_num = 110
        self.cam_num = 18

        self.is_train = (sub_folder == 'train')

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))

            # file_path = os.path.join(self.path, 'landmarks', self.selected_keys[num_i])
            # self.hdfs_lmk[num_i] = h5py.File(file_path, 'r', swmr=True)
            # assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)]
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None 

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.hdf_lmk = None
        self.transform = transform

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode
        
        data = {}
        data['mirror'] = 1.

        sub_id = int(self.selected_keys[key][7:11])
        data['sub_id'] = sub_id

        # Get face image
        image = self.hdf['face_patch'][idx, :].copy()
        data['image'] = image

        cam_id = int(self.hdf['cam_index'][idx][0])
        data['cam_id'] = cam_id - 1

        self.hdf_lmk = h5py.File(os.path.join(self.path, 'landmarks', self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf_lmk.swmr_mode
        lmk = self.hdf_lmk['landmark'][idx].copy()

        left_eye_box = get_rect(lmk[42:47], scale=1.)
        right_eye_box = get_rect(lmk[36:41], scale=1.)

        data['left_eye_box'] = left_eye_box
        data['right_eye_box'] = right_eye_box

        # Get labels
        if self.is_train or self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :].copy()
            gaze_label = gaze_label.astype('float')
            data['gaze'] = gaze_label

        if True:
            head_pose = self.hdf['face_head_pose'][idx, :].copy()
            data['pose'] = head_pose

        if self.transform is not None:
            data = self.transform(data)

        return data


def get_rect(points, ratio=1.0, scale=1):  # ratio = w:h
    x = points[:, 0]
    y = points[:, 1]

    x_expand = 0.1 * (max(x) - min(x))
    y_expand = 0.1 * (max(y) - min(y))

    x_max, x_min = max(x) + x_expand, min(x) - x_expand
    y_max, y_min = max(y) + y_expand, min(y) - y_expand

    # h:w=1:2
    if (y_max - y_min) * ratio < (x_max - x_min):
        h = (x_max - x_min) / ratio
        pad = (h - (y_max - y_min)) / 2
        y_max += pad
        y_min -= pad
    else:
        h = (y_max - y_min)
        pad = (h * ratio - (x_max - x_min)) / 2
        x_max += pad
        x_min -= pad

    int(x_min), int(x_max), int(y_min), int(y_max)
    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    bbox = np.array(bbox)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (224*scale, 224*scale))
    rect = np.concatenate([aSrc, bSrc])

    return rect

def xgaze_dataset(data_dir, transform=None, is_train=True):

    # load dataset

    refer_list_file = os.path.join(data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    if is_train:
        sub_folder_use = 'train'
        is_shuffle = True
        is_load_label = True
    else:
        sub_folder_use = 'test'
        is_shuffle = False
        is_load_label = False

    dataset = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=transform, is_shuffle=is_shuffle, is_load_label=True)

    return dataset

if __name__ == '__main__':

    pass
import os
import h5py
from numpy.lib.npyio import load
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from datasets.data_transforms import create_transform
from utils import pitchyaw_to_vector_one, vector_to_pitchyaw_one

class OneData(Dataset):
    def __init__(self, data_dir, input_size, data_type='mpii', test_ids=[], is_train=True, transform=None, eth_test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.lab_path = os.path.join(data_dir, data_type+'.label')
        self.eth_test = eth_test
        if eth_test:
            self.lab_path = os.path.join(data_dir, 'eth_test_order.label')
        elif data_type == 'eth':
            if is_train:
                self.lab_path = os.path.join(data_dir, 'eth_train.label')
            else:
                self.lab_path = os.path.join(data_dir, 'eth_valid.label')
                # self.lab_path = os.path.join(data_dir, 'eth_valid_sampled.label')
        self.test_ids = test_ids
        self.is_train = is_train
        self.input_size = input_size

        with open(self.lab_path) as f:
            lines = f.read().splitlines()
        
        if len(test_ids) > 0:
            self.lines = self.get_lines(lines)
        else:
            self.lines = lines

        self.type = data_type
    
    def get_lines(self, lines):
        train_lines = []
        test_lines = []

        for line in lines:
            items = line.strip().split(' ')
            if int(items[1]) in self.test_ids:
                test_lines.append(line)
            else:
                train_lines.append(line)

        if self.is_train:
            return train_lines

        return test_lines

    def decode(self, line):
        items = line.strip().split(' ')
        tw, th = self.input_size

        data = {}
        data['type'] = 0 if self.type == 'eth' else 1
        data['subid'] = int(items[1])
        # data['path'] = items[2]
        # data['image'] = Image.open(os.path.join(self.data_dir, items[2]))
        data['image'] = cv2.imread(os.path.join(self.data_dir, items[2]))

        # if self.type != 'eth' or self.is_train:
        if not self.eth_test:
            data['gaze'] = np.array(items[3].split(",")).astype("float")
            data['pose'] = np.array(items[4].split(",")).astype("float")
            lmk = np.array(items[5].split(",")).astype("float").reshape((-1, 2))
        else:
            data['pose'] = np.array(items[3].split(",")).astype("float")
            lmk = np.array(items[4].split(",")).astype("float").reshape((-1, 2))

        # lmk = lmk * 1.0 * tw / 224

        left_eye_box = get_rect(lmk[42:47])
        right_eye_box = get_rect(lmk[36:41])
        data['left_eye_box'] = left_eye_box
        data['right_eye_box'] = right_eye_box


        # data['gaze3d'] = pitchyaw_to_vector_one(data['gaze'])
        # data['gaze2d'] = vector_to_pitchyaw_one(data['gaze3d'])

        return data

    def __getitem__(self, index):
        
        line = self.lines[index]

        data = self.decode(line)

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):

        return len(self.lines)


def get_rect(points, ratio=1.0, size=224):  # ratio = w:h
    x = points[:, 0]
    y = points[:, 1]

    x_expand = 0.1 * (np.max(x) - np.min(x))
    y_expand = 0.1 * (np.max(y) - np.min(y))

    x_max, x_min = np.max(x) + x_expand, np.min(x) - x_expand
    y_max, y_min = np.max(y) + y_expand, np.min(y) - y_expand

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
    bSrc = np.minimum(bbox[:2] + bbox[2:], (size, size))
    rect = np.concatenate([aSrc, bSrc])

    return rect

def create_one_loader(data_dir, input_size, batch_size, data_type='diap', test_ids=[], is_train=False, eth_test=False):
    is_train = is_train and (not eth_test)
    transform = create_transform(input_size, is_train=is_train)
    one_data = OneData(data_dir=data_dir, input_size=input_size, data_type=data_type, 
        test_ids=test_ids, is_train=is_train, transform=transform, eth_test=eth_test)

    loader = torch.utils.data.DataLoader(
        one_data,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True
    )

    return loader



if __name__ == '__main__':

    onedata = OneData(data_dir='./data/GazeSetsPng', input_size=[224, 224], data_type='mpii', test_ids=[0, 1, 2])

    print(len(onedata))

    for i, data in enumerate(onedata):
        # if i % 10000 != 0:
        #     continue

        print(i, data['type'], data['subid'], data['gaze3d'], data['gaze'], data['gaze2d'])
        # cv2.imshow("image", data['image'])
        # cv2.waitKey(0)
    

    # loader = create_one_loader('./data/GazeSetsPng/', (224, 224), 64, 'eth', [], is_train=False)
    # print(len(loader), len(loader.dataset))
    # gazes = []

    # for i, data in enumerate(loader):
    #     pass

    # gazes = torch.cat(gazes).numpy()

    # gazes = gazes / np.pi * 180

    # print(gazes.shape)


    # from matplotlib import pyplot as plt
    # H, xedges, yedges = np.histogram2d(gazes[:, 1], gazes[:, 0], bins=100, range=[[-80, 80], [-80, 80]])
    # plt.imshow(H.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], interpolation='nearest', origin='lower')
    # plt.show()
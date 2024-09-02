import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image


class AllData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.lab_path = os.path.join(data_dir, 'all.label')
        self.transform = transform

        with open(self.lab_path) as f:
            self.lines = f.read().splitlines()

        self.types = ['mpii', 'diap', 'eth']

    def decode(self, line):
        items = line.strip().split(' ')

        data = {}
        data['type'] = self.types.index(items[0])
        data['subid'] = int(items[1])
        # data['path'] = items[2]
        # data['image'] = Image.open(os.path.join(self.data_dir, items[2]))
        data['image'] = cv2.imread(os.path.join(self.data_dir, items[2]))
        data['gaze'] = np.array(items[3].split(",")).astype("float")
        data['pose'] = np.array(items[4].split(",")).astype("float")
        lmk = np.array(items[5].split(",")).astype("float").reshape((-1, 2))

        left_eye_box = get_rect(lmk[42:47], scale=1.)
        right_eye_box = get_rect(lmk[36:41], scale=1.)
        data['left_eye_box'] = left_eye_box
        data['right_eye_box'] = right_eye_box

        return data

    def __getitem__(self, index):
        
        line = self.lines[index]

        data = self.decode(line)

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):

        return len(self.lines)


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


if __name__ == '__main__':

    alldata = AllData(data_dir='/data/GazeSets')

    print(len(alldata))

    for i, data in enumerate(alldata):
        if i % 10000 != 0:
            continue

        print(i, data['type'], data['subid'], data['path'], data['gaze'])
        cv2.imshow("image", data['face'])
        cv2.waitKey(0)
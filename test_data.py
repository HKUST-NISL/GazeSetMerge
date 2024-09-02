import torch
from trainer import Trainer
from itertools import cycle
import numpy as np

from utils import load_config, save_config
from datasets import create_gaze_dataloader


config = load_config()

# xgaze_data = create_gaze_dataloader(config, 'eth')
mpii_data = create_gaze_dataloader(config, 'mpii', test_ids=[0])


# print(len(xgaze_data), len(mpii_data))
# print(len(zip(cycle(xgaze_data), mpii_data)))

# for i, (data1, data2) in enumerate(zip(cycle(xgaze_data), mpii_data)):
#     print(i)


for i, data in enumerate(mpii_data):
    print(i)

import os
import numpy as np
from tqdm import tqdm

eth_path = 'data/GazeSetsPng/eth.label'
train_path = 'data/GazeSetsPng/eth_train.label'
valid_path = 'data/GazeSetsPng/eth_valid.label'

valid_ids = [3, 32, 33, 48, 52, 62, 80, 88, 101, 109]
with open(eth_path) as f:
    lines = f.read().splitlines()

train_f = open(train_path, 'w')
valid_f = open(valid_path, 'w')

for i in tqdm(range(len(lines))):
    line = lines[i]
    str_splits = line.split(' ')
    sub_id = int(str_splits[1])
    if sub_id in valid_ids:
        valid_f.write(line+'\n')
    else:
        train_f.write(line+'\n')

train_f.close()
valid_f.close()
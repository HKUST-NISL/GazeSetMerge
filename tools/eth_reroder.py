import os
import numpy
import json

org_path = './data/GazeSetsPng/eth_test.label'
ord_path = './data/GazeSetsPng/eth_test_order.label'

eth_list_file = './data/GazeSetsPng/ETH-Gaze/train_test_split.json'

with open(eth_list_file, 'r') as f:
    datastore = json.load(f)

test_list = datastore['test']

with open(org_path) as f:
    org_lines = f.readlines()


with open(ord_path, 'w') as f:
    for sub in test_list:
        sub_name = sub[:-3]
        print(sub_name)

        for line in org_lines:
            if sub_name in line:
                f.write(line)

print(ord_path)
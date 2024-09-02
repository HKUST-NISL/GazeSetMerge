#!/bin/bash
for i in {0..14}
do
python main.py --config=$1 batch_size 64 test_ids "[${i}]" gpu_id -1
done
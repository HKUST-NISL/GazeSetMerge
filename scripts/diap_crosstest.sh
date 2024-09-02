#!/bin/bash
python test_gaze.py --config=$1 batch_size 64 test_ids '[0, 1, 2, 3]' test_data $2 train_tag $3
python test_gaze.py --config=$1 batch_size 64 test_ids '[4, 5, 6, 7]' test_data $2 train_tag $3
python test_gaze.py --config=$1 batch_size 64 test_ids '[8, 9, 10, 11]' test_data $2 train_tag $3
python test_gaze.py --config=$1 batch_size 64 test_ids '[12, 13, 14, 15]' test_data $2 train_tag $3
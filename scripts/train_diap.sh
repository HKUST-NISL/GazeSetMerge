#!/bin/bash
DATEtIME="t`date +"%Y%m%d%H%M"`"

python main.py --config=$1 batch_size 64 test_ids '[0, 1, 2, 3]' train_tag ${DATEtIME}
python main.py --config=$1 batch_size 64 test_ids '[4, 5, 6, 7]' train_tag ${DATEtIME}
python main.py --config=$1 batch_size 64 test_ids '[8, 9, 10, 11]' train_tag ${DATEtIME}
python main.py --config=$1 batch_size 64 test_ids '[12, 13, 14, 15]' train_tag ${DATEtIME}
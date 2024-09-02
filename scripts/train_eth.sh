#!/bin/bash
DATEtIME="t`date +"%Y%m%d%H%M"`"

python main.py --config=$1 batch_size 50 train_tag ${DATEtIME}
python main.py --config=$1 is_train False train_tag ${DATEtIME}


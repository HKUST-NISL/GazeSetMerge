#!/bin/bash
DATEtIME="t`date +"%Y%m%d%H%M"`"

for i in {0..14}
do
python main.py --config=$1 batch_size 64 test_ids "[${i}]" train_tag ${DATEtIME}
done
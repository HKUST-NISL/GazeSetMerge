#!/bin/bash
for i in {0..14}
do
python test_gaze.py --config=$1 test_ids "[${i}]" test_data $2 train_tag $3
done
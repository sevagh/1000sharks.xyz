#!/usr/bin/env bash

# leave this one overnight to train

python train.py --id periphery_only --data_dir ./experiment-1/periphery-chunks/ --num_epochs 100 --batch_size 64 --sample_rate 16000

python train.py --id mestis_only --data_dir ./experiment-1/mestis-chunks/ --num_epochs 100 --batch_size 64 --sample_rate 16000

python train.py --id mestiphery --data_dir ./experiment-1/mixed-chunks/ --num_epochs 100 --batch_size 64 --sample_rate 16000

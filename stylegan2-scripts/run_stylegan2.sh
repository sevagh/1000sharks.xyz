#!/usr/bin/env bash

# make sure you have enough shark and metal album cover images in those two directories
python crop_images.py --dim=256 ./output-images/ ./shark-images/ ./metal-album-covers/

python dataset_tool.py create_from_images datasets/1000sharks/ ./output-images/

# ~31 hours of training on an RTX 2070 SUPER
python run_training.py --data-dir=./datasets/ --dataset=1000sharks --config=config-e --total-kimg=1000

# generate 1000 images
python run_generator.py generate-images --seeds=0-999 --truncation-psi=1.0 --network=results/00008-stylegan2-1000sharks-1gpu-config-e/network-final.pkl

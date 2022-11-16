#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=3

python3 ./tools/train_test3.py --dataset ycb\
  --dataset_root ~/Instance/instance_module/datasets/ycb/YCB_Video_Dataset

#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=3

python3 ./tools/train_test3.py --dataset linemod\
  --dataset_root ./datasets/linemod/Linemod_preprocessed

#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


python3 ./tools/eval_ycb_icp.py --dataset_root ~/PVN3D/pvn3d/datasets/ycb/YCB_Video_Dataset\
  --model trained_models/ycb/test3/pose_model_23_0.017398578480653314.pth\
  --seg_model trained_models/ycb/test3/seg_0.08592522922650737.pth\


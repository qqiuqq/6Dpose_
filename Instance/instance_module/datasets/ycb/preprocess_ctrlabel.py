#!/usr/bin/env python3
import os
import cv2
import tqdm
import torch
import os.path
import numpy as np
from common import Config
import pickle as pkl
from lib.utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
from datasets.ycb.ycb_ctr_label import YCB_Dataset

config = Config(dataset_name='ycb')
bs_utils = Basic_Utils(config)
torch.multiprocessing.set_sharing_strategy('file_system')

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():

    ctr_label_ds = YCB_Dataset('test')
    ctr_label_loader = torch.utils.data.DataLoader(
        ctr_label_ds, batch_size=config.test_mini_batch_size, shuffle=False,
        num_workers=16, worker_init_fn=worker_init_fn
    )

    for i, data in tqdm.tqdm(
        enumerate(ctr_label_loader), leave=False, desc='Preprocessing ctr_label'
    ):
        ctr = data




if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab

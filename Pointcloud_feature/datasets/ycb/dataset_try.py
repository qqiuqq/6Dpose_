import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
# from common import Config
#
# config = Config(dataset_name='ycb')


def read_lines(p):
    with open(p, 'r') as f:
        lines = [
            line.strip() for line in f.readlines()
        ]
    return lines


class PoseDataset(data.Dataset):
    def __init__(self, mode, refine):
        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
        else:
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'

        self.cls_lst = read_lines(os.path.join('datasets/ycb/dataset_config/classes.txt'))
        self.ycb_cls_ctr_dict = {}
        self.mode = mode
        self.list = []
        self.real = []
        self.syn = []
        self.refine = refine
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_file = open('datasets/ycb/dataset_config/classes.txt')
        class_id = 1
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('{0}/models/{1}/points.xyz'.format('datasets/ycb/YCB_Video_Dataset', class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()

            class_id += 1

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.front_num = 2

        print(len(self.list))

    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format('datasets/ycb/YCB_Video_Dataset', self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format('datasets/ycb/YCB_Video_Dataset', self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format('datasets/ycb/YCB_Video_Dataset', self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format('datasets/ycb/YCB_Video_Dataset', self.list[index]))

        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        # if self.mode == 'train':
        #     for k in range(5):
        #         seed = random.choice(self.syn)
        #         front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format('datasets/ycb/YCB_Video_Dataset', seed)).convert("RGB")))
        #         front = np.transpose(front, (2, 0, 1))
        #         f_label = np.array(Image.open('{0}/{1}-label.png'.format('datasets/ycb/YCB_Video_Dataset', seed)))
        #         front_label = np.unique(f_label).tolist()[1:]
        #         if len(front_label) < self.front_num:
        #             continue
        #         front_label = random.sample(front_label, self.front_num)
        #         for f_i in front_label:
        #             mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
        #             if f_i == front_label[0]:
        #                 mask_front = mk
        #             else:
        #                 mask_front = mask_front * mk
        #         t_label = label * mask_front
        #         if len(t_label.nonzero()[0]) > 1000:
        #             label = t_label
        #             add_front = True
        #             break

        obj = meta['cls_indexes'].flatten().astype(np.int32)
        count = 0
        while 1:
            count += 1
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > 500:
                break

        if self.mode == 'train':
            img = self.trancolor(img)

        msk_dp = depth > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        label = label.flatten()[choose]
        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy

        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        choose = np.array([choose])
        choose_2 = np.array([i for i in range(len(choose[0, :]))])

        if len(choose_2) < 400:
            return None
        if len(choose_2) > 30000:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:30000] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, 30000 - len(choose_2)), 'wrap')

        cloud = cloud[choose_2, :]
        choose = choose[:, choose_2]
        label = label[choose_2].astype(np.int32)

        # 读取的是整张图像
        # rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))
        # 虚拟图像的背景加入真实背景值
        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(
                Image.open('{0}/{1}-color.png'.format('datasets/ycb/YCB_Video_Dataset', seed)).convert("RGB")))
            back = np.transpose(back, (2, 0, 1))
            img_masked = back * mask_back + img
        else:
            img_masked = img
        # 增加噪声也是针对整张图像
        if self.mode == 'train' and add_front:
            img_masked = img_masked * mask_front + front * ~(mask_front)

        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        add_t = np.array([random.uniform(-0.03, 0.03) for i in range(3)])

        if self.mode == 'train':
            cloud = np.add(cloud, add_t)

        ctr3ds = np.zeros((22, 3))
        RTs = np.zeros((22, 3, 4))
        ctr_targ_ofst = np.zeros((30000, 3))
        for i, cls_id in enumerate(obj):
            r = meta['poses'][:, :, i][:, 0:3]
            t = np.array(meta['poses'][:, :, i][:, 3:4].flatten()[:, None])
            RT = np.concatenate((r, t), axis=1)
            RTs[i] = RT

            ctr = self.get_ctr(self.cls_lst[cls_id - 1]).copy()[:, None]
            ctr = np.dot(ctr.T, r.T) + t[:, 0]
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(label == cls_id)[0]
            target_offset = np.array(np.add(cloud, -1.0 * t.T))
            ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        add_t = np.array([random.uniform(-0.03, 0.03) for i in range(3)])
        dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        dellist = random.sample(dellist, len(self.cld[obj[idx]]) - 2600)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)
        target = np.dot(model_points, target_r.T)
        if True:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)
        return self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               torch.LongTensor(label.astype(np.int32)), \
               torch.LongTensor([int(obj[idx]) - 1]), \
               torch.from_numpy(ctr_targ_ofst.astype(np.float32)), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32))

    def get_ctr(self, cls, ds_type='ycb', ctr_pth=None):
        if ctr_pth:
            return np.loadtxt(ctr_pth)
        if type(cls) is int:
            if ds_type == 'ycb':
                cls = self.ycb_cls_lst[cls - 1]
            else:
                cls = self.config.lm_id2obj_dict[cls]
        if ds_type == "ycb":
            if cls in self.ycb_cls_ctr_dict.keys():
                return self.ycb_cls_ctr_dict[cls].copy()
            cor_pattern = os.path.join(
                'datasets/ycb/ycb_object_kps', '{}/corners.txt'.format(cls),
            )
            cors = np.loadtxt(cor_pattern.format(cls), dtype=np.float32)
            ctr = cors.mean(0)
            self.ycb_cls_ctr_dict[cls] = ctr
        return ctr.copy()

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_large




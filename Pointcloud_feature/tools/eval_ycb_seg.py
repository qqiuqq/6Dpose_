import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network_pointnet3 import PoseNet#, PoseRefineNet
from lib.testnew1 import SEG
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from kmeans_pytorch import kmeans
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--seg_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 500
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/ycb-pointnet/Densefusion_wo_refine_result1'
result_refine_dir = 'experiments/eval_result/ycb-pointnet/Densefusion_iterative_result1'
ycb_r_lst = torch.tensor([[0.07005500048398972],
[0.10672450065612793],
[0.08801200240850449],
[0.050907500088214874],
[0.09565199911594391],
[0.042785000056028366],
[0.0689530000090599],
[0.05055350065231323],
[0.05105699971318245],
[0.0892150029540062],
[0.12116599828004837],
[0.12529299780726433],
[0.08063450083136559],
[0.05845149792730808],
[0.09373300150036812],
[0.10294999927282333],
[0.10074950009584427],
[0.06046199984848499],
[0.0855565033853054],
[0.10487799718976021],
[0.0389384999871254]]).cuda()


def cluster_mask(
    pcld, mask, ctr_of):
    pred_ctr = pcld[0] - ctr_of[0][0]
    _, n_pts, _ = ctr_of[0].size()
    pred_cls_ids = np.unique(mask[mask > 0].contiguous().cpu().numpy())
    ctrs = []
    radius = 0.08
    for icls, cls_id in enumerate(pred_cls_ids):
        if (cls_id == 20 and 19 in pred_cls_ids):
            continue
        if (cls_id == 19 and 20 in pred_cls_ids):
            cls_msk = (mask == 19) | (mask == 20)
            cluster_ids_x, cluster_centers = kmeans(
                X=pred_ctr[cls_msk, :], num_clusters=2, distance='euclidean', device=torch.device('cuda:0')
            )
            ctrs.append(cluster_centers[1].detach().contiguous().cpu().numpy())
            ctrs.append(cluster_centers[0].detach().contiguous().cpu().numpy())
            continue
        cls_msk = (mask == cls_id)
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        ctrs.append(ctr.detach().contiguous().cpu().numpy())
    # print(ctrs)
    ctrs = torch.from_numpy(np.array(ctrs).astype(np.float32)).cuda()
    n_ctrs, _ = ctrs.size()
    pred_ctr_rp = pred_ctr.view(n_pts, 1, 3).repeat(1, n_ctrs, 1)
    ctrs_rp = ctrs.view(1, n_ctrs, 3).repeat(n_pts, 1, 1)
    ctr_dis = torch.norm((pred_ctr_rp - ctrs_rp), dim=2)
    min_dis, min_idx = torch.min(ctr_dis, dim=1)
    msk_closest_ctr = torch.LongTensor(pred_cls_ids).cuda()[min_idx]
    new_msk = mask.clone()
    for cls_id in pred_cls_ids:
        if cls_id == 20 and 19 in pred_cls_ids:
            continue
        if cls_id == 19 and 20 in pred_cls_ids:
            min_msk_19 = min_dis < ycb_r_lst[cls_id - 1] * 0.8
            min_msk_20 = min_dis < ycb_r_lst[cls_id] * 0.8
            update_msk_19 = (mask > 0) & (msk_closest_ctr == cls_id) & min_msk_19
            update_msk_20 = (mask > 0) & (msk_closest_ctr == cls_id + 1) & min_msk_20
            pcld_mask = pcld[0][update_msk_19]
            N, c = pcld_mask.size()
            Ar = pcld_mask.view(1, N, c).repeat(N, 1, 1)
            Cr = pcld_mask.view(N, 1, c).repeat(1, N, 1)
            dis = torch.norm(Cr - Ar, dim=2)
            dis_max = torch.max(dis[dis < 0.5])
            if dis_max > 0.19:
                new_msk[update_msk_19] = msk_closest_ctr[update_msk_19] + 1
                new_msk[update_msk_20] = msk_closest_ctr[update_msk_20] - 1
            else:
                new_msk[update_msk_19] = msk_closest_ctr[update_msk_19]
                new_msk[update_msk_20] = msk_closest_ctr[update_msk_20]
            continue
        if cls_id == 17:
            update_msk = (mask == cls_id)
            new_msk[update_msk] = mask[update_msk]
        else:
            min_msk = min_dis < ycb_r_lst[cls_id - 1] * 0.8
            update_msk = (mask > 0) & (msk_closest_ctr == cls_id) & min_msk
            new_msk[update_msk] = msk_closest_ctr[update_msk]
    mask = new_msk
    return mask, ctrs


def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * torch.sqrt(2 * torch.tensor(np.pi)))) \
        * torch.exp(-0.5 * ((distance / bandwidth)) ** 2)
class MeanShiftTorch():
    def __init__(self, bandwidth=0.05, max_iter=300):
        self.bandwidth = bandwidth
        self.stop_thresh = bandwidth * 1e-3
        self.max_iter = max_iter

    def fit(self, A):
        """
        params: A: [N, 3]
        """
        N, c = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            Ar = A.view(1, N, c).repeat(N, 1, 1)
            Cr = C.view(N, 1, c).repeat(1, N, 1)
            dis = torch.norm(Cr - Ar, dim=2)
            w = gaussian_kernel(dis, self.bandwidth).view(N, N, 1)
            new_C = torch.sum(w * Ar, dim=1) / torch.sum(w, dim=1)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=1)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = A.view(N, 1, c).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=2)
        num_in = torch.sum(dis < self.bandwidth, dim=1)
        max_num, max_idx = torch.max(num_in, 0)
        labels = dis[max_idx] < self.bandwidth
        return C[max_idx, :], labels
estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()
seg_estimator = SEG(num_classes=22, num_points=30000)
seg_estimator = torch.nn.DataParallel(seg_estimator)
seg_estimator.cuda()
seg_estimator.load_state_dict(torch.load(opt.seg_model))
seg_estimator.eval()
"""
refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()
"""
testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1

for now in range(2142, 2949):
    img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
    label = np.array(Image.open('{0}/{1}-label.png'.format(opt.dataset_root, testlist[now])))
    meta = scio.loadmat('{0}/{1}-meta.mat'.format(opt.dataset_root, testlist[now]))
    #posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
    #label = np.array(posecnn_meta['labels'])
    #posecnn_rois = np.array(posecnn_meta['rois'])
    #lst = posecnn_rois[:, 1:2].flatten()
    lst = meta['cls_indexes'].flatten().astype(np.int32)
    my_result_wo_refine = []
    my_result = []

    msk_dp = depth > 1e-6
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
    label = label.flatten()[choose]
    depth = depth.flatten()[choose]
    xmap_mask = xmap.flatten()[choose]
    ymap_mask = ymap.flatten()[choose]
    choose = np.array([choose])

    choose_2 = np.array([i for i in range(len(choose[0, :]))])
    if len(choose_2) > 30000:
        c_mask = np.zeros(len(choose_2), dtype=int)
        c_mask[:30000] = 1
        np.random.shuffle(c_mask)
        choose_2 = choose_2[c_mask.nonzero()]
    else:
        choose_2 = np.pad(choose_2, (0, 300000 - len(choose_2)), 'wrap')
    choose = choose[:, choose_2]

    choose = np.array([choose])
    choose = torch.LongTensor(choose.astype(np.int32))
    label = label[choose_2].astype(np.int32)
    depth = depth.flatten()[choose_2]
    xmap_mask = xmap_mask.flatten()[choose_2]
    ymap_mask = ymap_mask.flatten()[choose_2]

    pt2 = depth[:, np.newaxis] / cam_scale
    pt0 = (ymap_mask[:, np.newaxis] - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mask[:, np.newaxis] - cam_cy) * pt2 / cam_fy
    pcld = np.concatenate((pt0, pt1, pt2), axis=1)
    pcld = torch.from_numpy(np.array([pcld]).astype(np.float32)).cuda()
    img_masked = np.array(img)[:, :, :3]
    img_masked = np.transpose(img_masked, (2, 0, 1))

    img_masked = norm(torch.from_numpy(img_masked.astype(np.float32))).view(1, 3, 480, 640)
    pred_rgbd_seg, pred_ctr = seg_estimator(img_masked, pcld, choose)
    #pred_rgbd_seg = seg_estimator(img_masked, choose)
    _, classes_rgbd = torch.max(pred_rgbd_seg[0], -1)
    mask, ctrs = cluster_mask(pcld, classes_rgbd, pred_ctr)
    print(classes_rgbd)
    print(mask)
    for idx in range(len(lst)):
        itemid = lst[idx]
        try:
            # rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(mask.cpu().numpy(), itemid))
            print(itemid)
            # print(torch.sum((torch.LongTensor(label.astype(np.int32)).cuda() == itemid)))
            # print(torch.sum((mask == itemid)))

            # print(label.device, classes_rgbd.device, itemid.device)
            print("before:", len(((torch.LongTensor(label.astype(np.int32)).cuda() == itemid) * (
                    classes_rgbd == itemid)).nonzero()) / torch.sum(
                classes_rgbd == itemid))
            print("after:", len(((torch.LongTensor(label.astype(np.int32)).cuda() == itemid) * (
                        mask == itemid)).nonzero()) / torch.sum(
                mask == itemid))
            # mask = mask_label * mask_depth

            choose = mask_label.flatten().nonzero()[0]
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            elif len(choose) == 0:
                my_result_wo_refine.append([0.0 for i in range(7)])
                my_result.append([0.0 for i in range(7)])
                continue
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

            depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap_mask.flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap_mask.flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([itemid - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])
            pred_r, pred_t, pred_c = estimator(cloud, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)
            # print("dis1:", np.linalg.norm(gt_t - ctr))
            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            # print('dis2:', np.linalg.norm(gt_t - my_t.reshape(3,1)))
            my_pred = np.append(my_r, my_t)
            my_result_wo_refine.append(my_pred.tolist())
            """
            for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = refiner(new_cloud, emb, index)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)

                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final

            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
            """
            my_result.append(my_pred.tolist())
        except ZeroDivisionError:
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])

    scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, '%04d' % now), {'poses':my_result_wo_refine})
    scio.savemat('{0}/{1}.mat'.format(result_refine_dir, '%04d' % now), {'poses':my_result})
    print("Finish No.{0} keyframe".format(now))


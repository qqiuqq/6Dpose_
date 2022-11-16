import _init_paths
import argparse
import os
import random
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset_test import PoseDataset as PoseDataset_linemod
from lib.network_pointnet3 import PoseNet
from lib.loss_mul import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
from lib.util.icp.icp import my_icp, best_fit_transform
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
iteration = 4
bs = 1
dataset_config_dir = 'datasets/linemod/dataset_config'
output_result_dir = 'experiments/eval_result/linemod'
knn = KNearestNeighbor(1)

estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
# refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
# refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
# refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
# refiner.eval()

testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(diameter)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

for i, data in enumerate(testdataloader, 0):
    img, points, rgb_cld, choose, label, target, model_points, idx = data
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    img, points, rgb_cld, choose, label, target, model_points, idx = Variable(img).cuda(), \
                                                                     Variable(points).cuda(), \
                                                                     Variable(rgb_cld).cuda(), \
                                                                     Variable(choose).cuda(), \
                                                                     Variable(label).cuda(), \
                                                                     Variable(target).cuda(), \
                                                                     Variable(model_points).cuda(), \
                                                                     Variable(idx).cuda()

    points = np.array(points.cpu())
    rgb_cld = np.array(rgb_cld.cpu())
    if len(points[0, :, 0]) > 500:
        c_mask = np.zeros(len(points[0, :, 0]), dtype=int)
        c_mask[:500] = 1
        np.random.shuffle(c_mask)
        points = points[:, c_mask.nonzero(), :][0]
        rgb_cld = rgb_cld[:, c_mask.nonzero(), :][0]

    points = torch.from_numpy(points.astype(np.float32)).cuda()
    rgb_cld = torch.from_numpy(rgb_cld.astype(np.float32)).cuda()

    pred_r, pred_t, pred_c = estimator(points, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)
    points = points.view(bs * num_points, 1, 3)
    # print("dis1:", np.linalg.norm(gt_t - ctr))
    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    # 加入icp算法
    model_points = model_points[0].cpu().detach().numpy()
    R = quaternion_matrix(my_r)[:3, :3]
    pred = np.dot(model_points, R.T) + my_t
    target = target[0].cpu().detach().numpy()

    icp_pose, dis, _ = my_icp(
        pred, target, init_pose=None,
        max_iterations=500,
        tolerance=1e-9
    )
    my_mat = quaternion_matrix(my_r)
    my_mat[0:3, 3] = my_t
    my_mat_2 = icp_pose
    my_mat_final = np.dot(my_mat_2, my_mat)
    my_r_final = copy.deepcopy(my_mat_final)
    my_r_final[0:3, 3] = 0
    my_r_final = quaternion_from_matrix(my_r_final, True)
    my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

    my_pred = np.append(my_r_final, my_t_final)
    my_r = my_r_final
    my_t = my_t_final
    my_r = quaternion_matrix(my_r)[:3, :3]
    pred = np.dot(model_points, my_r.T) + my_t
    # 不加入ICP时需要注释掉上面一段，使用以下四句
    # model_points = model_points[0].cpu().detach().numpy()
    # my_r = quaternion_matrix(my_r)[:3, :3]
    # pred = np.dot(model_points, my_r.T) + my_t
    # target = target[0].cpu().detach().numpy()
    if idx[0].item() in sym_list:
        pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
        inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
        target = torch.index_select(target, 1, inds.view(-1) - 1)
        dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    else:
        dis = np.mean(np.linalg.norm(pred - target, axis=1))

    if dis < diameter[idx[0].item()]:
        success_count[idx[0].item()] += 1
        print('No.{0} Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    else:
        print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    num_count[idx[0].item()] += 1

for i in range(num_objects):
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.close()

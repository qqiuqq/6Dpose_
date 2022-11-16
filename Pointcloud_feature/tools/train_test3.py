# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
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
from datasets.ycb.dataset_test import PoseDataset as PoseDataset_ycb
#from datasets.warehouse.dataset import PoseDataset as PoseDataset_warehouse
from datasets.linemod.dataset_test import PoseDataset as PoseDataset_linemod
from lib.network_pointnet3 import PoseNet # , PoseRefineNet
# from lib.testnew1 import SEG
import numpy.ma as ma
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# 由命令行转换数据
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or warehouse or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Warehouse_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--seg_model', type=str, default = 'seg_0.08423834332091323.pth',  help='resume PoseRefineNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

# 主函数开始
def main():
    opt.manualSeed = random.randint(1, 100)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    # 由datasets.ycb.dataset_test加载数据
    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 500 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb/test3' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'warehouse':
        opt.num_objects = 13
        opt.num_points = 1000
        opt.outf = 'trained_models/warehouse'
        opt.log_dir = 'experiments/logs/warehouse'
        opt.repeat_epoch = 1
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return
    # 由lib.network_pointnet3 加载模型
    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()
    # 可选择地是否载入已有模型数据
    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
        opt.refine_start = False
        opt.decay_start = False
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
    # 加载数据的不同调用函数
    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, False, opt.dataset_root, opt.noise_trans, opt.refine_start)
        # dataset = PoseDataset_ycb('train', opt.refine_start)
    elif opt.dataset == 'warehouse':
        dataset = PoseDataset_warehouse('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
        # test_dataset = PoseDataset_ycb('test', opt.refine_start)
    elif opt.dataset == 'warehouse':
        test_dataset = PoseDataset_warehouse('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    # 数据加载完成开始训练
    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))
    # 损失函数调用
    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    dis_train = []
    loss_train = []
    dis_val = []
    loss_val = []
    # 开始一个epoch
    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis = 0.0
        train_dis_avg = 0.0
        train_loss_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()
        # 开始每个batch的训练
        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                # 加载数据
                img, points, rgb_cld, choose, label, target, model_points, idx, ctr_ofs = data
                img, points, rgb_cld, choose, label, target, model_points, idx, ctr_ofs = Variable(img).cuda(), \
                                                                                          Variable(points).cuda(), \
                                                                                          Variable(rgb_cld).cuda(), \
                                                                                          Variable(choose).cuda(), \
                                                                                          Variable(label).cuda(), \
                                                                                          Variable(target).cuda(), \
                                                                                          Variable(model_points).cuda(), \
                                                                                          Variable(idx).cuda(), \
                                                                                          Variable(ctr_ofs).cuda()

                # 采用500个点
                points = np.array(points.cpu())
                choose = np.array(choose.cpu())
                if len(points[0, :, 0]) > 500:
                    c_mask = np.zeros(len(ctr_ofs[0, :, 0]), dtype=int)
                    c_mask[:500] = 1
                    np.random.shuffle(c_mask)
                    points = points[:, c_mask.nonzero(), :][0]
                    choose = choose[:, c_mask.nonzero()][0]

                points = torch.from_numpy(points.astype(np.float32)).cuda()
                choose = torch.from_numpy(choose.astype(np.int64)).cuda()
                # 数据输入模型得到输出
                pred_r, pred_t, pred_c = estimator(points, points, choose, idx)
                # 调用loss函数计算loss值
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points,
                                                              opt.w, opt.refine_start)

                loss.backward()

                train_dis_avg += dis.item()
                train_dis += dis.item()
                train_loss_avg += loss.item()
                train_count += 1

                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0.0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        dis_train.append(train_dis / train_count)
        loss_train.append(train_loss_avg / train_count)
        # 训练完一个epoch进行测试
        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_loss = 0.0
        test_count = 0
        estimator.eval()
        # refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            img, points, rgb_cld, choose, label, target, model_points, idx, ctr_ofs = data
            img, points, rgb_cld, choose, label, target, model_points, idx, ctr_ofs = Variable(img).cuda(), \
                                                                                      Variable(points).cuda(), \
                                                                                      Variable(rgb_cld).cuda(), \
                                                                                      Variable(choose).cuda(), \
                                                                                      Variable(label).cuda(), \
                                                                                      Variable(target).cuda(), \
                                                                                      Variable(model_points).cuda(), \
                                                                                      Variable(idx).cuda(), \
                                                                                      Variable(ctr_ofs).cuda()

            points = np.array(points.cpu())
            choose = np.array(choose.cpu())
            if len(points[0, :, 0]) > 500:
                c_mask = np.zeros(len(ctr_ofs[0, :, 0]), dtype=int)
                c_mask[:500] = 1
                np.random.shuffle(c_mask)
                points = points[:, c_mask.nonzero(), :][0]
                choose = choose[:, c_mask.nonzero()][0]

            points = torch.from_numpy(points.astype(np.float32)).cuda()
            choose = torch.from_numpy(choose.astype(np.int64)).cuda()

            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points,
                                                          opt.w, opt.refine_start)


            test_dis += dis.item()
            test_loss += loss.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss))

            test_count += 1

        test_dis = test_dis / test_count
        test_loss = test_loss / test_count
        dis_val.append(test_dis)
        loss_val.append(test_loss)

        # 绘制曲线
        x = range(1, len(dis_val) + 1)
        plt.subplot(2, 1, 1)
        plt.plot(x, dis_train, 'o-', label='dis_train')
        plt.plot(x, dis_val, '*-', label='dis_val')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(x, loss_train, 'o-', label='loss_train')
        plt.plot(x, loss_val, '*-', label='loss_val')
        plt.legend()
        plt.savefig("trained_models/ycb/test3/curve.jpg")
        plt.close()

        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_loss))
        if test_loss <= best_test:
            best_test = test_loss
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_loss))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_loss))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
        # 如果测试所得loss值小于设定值，则将学习率调低
        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)


if __name__ == '__main__':
    main()

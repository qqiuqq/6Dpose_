import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet




class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.e_conv1 = torch.nn.Conv1d(6, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)

        self.conv5 = torch.nn.Conv1d(512, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, emb, x):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv3(x))
        emb = F.relu(self.e_conv3(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)  # 128 + 256 + 1024


class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat()
        self.featrefine = PointNetrefine()
        self.conv1_r = torch.nn.Conv1d(1792, 1280, 1)
        self.conv1_t = torch.nn.Conv1d(1792, 1280, 1)
        self.conv1_c = torch.nn.Conv1d(1792, 1280, 1)

        self.conv2_r = torch.nn.Conv1d(1280, 640, 1)
        self.conv2_t = torch.nn.Conv1d(1280, 640, 1)
        self.conv2_c = torch.nn.Conv1d(1280, 640, 1)

        self.conv3_r = torch.nn.Conv1d(640, 256, 1)
        self.conv3_t = torch.nn.Conv1d(640, 256, 1)
        self.conv3_c = torch.nn.Conv1d(640, 256, 1)

        self.conv4_r = torch.nn.Conv1d(256, 128, 1)
        self.conv4_t = torch.nn.Conv1d(256, 128, 1)
        self.conv4_c = torch.nn.Conv1d(256, 128, 1)

        self.conv5_r = torch.nn.Conv1d(128, num_obj * 4, 1)  # quaternion
        self.conv5_t = torch.nn.Conv1d(128, num_obj * 3, 1)  # translation
        self.conv5_c = torch.nn.Conv1d(128, num_obj * 1, 1)  # confidence
        self.dropout = nn.Dropout(p=0.5)
        self.num_obj = num_obj

    def forward(self, img, x, choose, obj):

        bs, di, _ = img.size()

        x = x.transpose(2, 1).contiguous()
        #emb = img.transpose(2, 1).contiguous()
        ap_x = self.feat(x)
        ap_x = self.featrefine(ap_x)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = F.relu(self.conv4_r(rx))
        tx = F.relu(self.conv4_t(tx))
        cx = F.relu(self.conv4_c(cx))

        rx = self.conv5_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv5_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv5_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        #b = 0
        #out_rx = torch.index_select(rx[b], 0, obj[b])
        #out_tx = torch.index_select(tx[b], 0, obj[b])
        #out_cx = torch.index_select(cx[b], 0, obj[b])
        rx_tensor = torch.zeros(bs, 4, self.num_points).cuda()
        tx_tensor = torch.zeros(bs, 3, self.num_points).cuda()
        cx_tensor = torch.zeros(bs, 1, self.num_points).cuda()
        for i in range(bs):
            out_rx = torch.index_select(rx[i], 0, obj[i])
            out_tx = torch.index_select(tx[i], 0, obj[i])
            out_cx = torch.index_select(cx[i], 0, obj[i])

            rx_tensor[i] = out_rx
            tx_tensor[i] = out_tx
            cx_tensor[i] = out_cx
        out_rx = rx_tensor.contiguous().transpose(2, 1).contiguous()
        out_tx = tx_tensor.contiguous().transpose(2, 1).contiguous()
        out_cx = cx_tensor.contiguous().transpose(2, 1).contiguous()

        #out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        #out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        #out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx

# 64维的通道权重
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.relu = nn.ReLU()

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x
        x = x.view(-1, self.k, 1)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, feature_weight=True):
        super(PointNetfeat, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(500)
        self.feature_weight = feature_weight
        if self.feature_weight:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]

        x = F.relu(self.conv1(x))

        if self.feature_weight:
            weight_feat = self.fstn(x)
            x = x * weight_feat
        else:
            weight_feat = None
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        pointfeat = x
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.ap1(x)

        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1)  # 1024 + 256

class PointNetrefine(nn.Module):
    def __init__(self):
        super(PointNetrefine, self).__init__()

        self.conv = torch.nn.Conv1d(1280, 1792, 1)
        self.conv1 = torch.nn.Conv1d(1280, 1280, 3, 1, 1)

        self.conv2 = torch.nn.Conv1d(1792, 1792, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        feature1 = F.relu(self.conv(x))
        feature2 = F.relu(self.conv(F.relu(self.conv1(x))))
        feature3 = F.relu(self.conv(F.relu(self.conv1(F.relu(self.conv1(x))))))
        feat = feature1 + feature2 + feature3
        x_weight = F.relu(self.conv2(feat))
        x_weight = torch.mean(x_weight, dim=1, keepdim=True)
        x_weight = self.sigmoid(x_weight)
        x = feat * x_weight

        return x

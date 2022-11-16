from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import lib.util.etw_pytorch_utils as pt_utils
from collections import namedtuple
from lib.pspnet import PSPNet, Modified_PSPNet
import torch.nn.functional as F
import numpy.ma as ma
import numpy as np
import math
import random
from torchvision import transforms

toPIL = transforms.ToPILImage()
psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

modified_psp_models = {
    'resnet18': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = modified_psp_models['resnet34'.lower()]()

    def forward(self, x):
        x, x_seg = self.model(x)
        return x, x_seg

class DenseFusion(nn.Module):
    def __init__(self, num_points):
        super(DenseFusion, self).__init__()
        self.conv1_cld = torch.nn.Conv1d(3, 128, 1)
        self.conv2_cld = torch.nn.Conv1d(128, 256, 1)

        self.conv2_rgb = torch.nn.Conv1d(128, 256, 1)

        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.n_pts = num_points

    def forward(self, rgb_emb, pcld_emb):
        cld1 = F.relu(self.conv1_cld(pcld_emb))

        feat_1 = torch.cat((rgb_emb, cld1), dim=1)
        rgb2 = F.relu(self.conv2_rgb(rgb_emb))
        cld2 = F.relu(self.conv2_cld(cld1))
        feat_2 = torch.cat((rgb2, cld2), dim=1)
        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.n_pts)
        return torch.cat([feat_1, feat_2, ap_x], 1) # 256 + 512 + 1024 = 1792


class SEG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        pcld_input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        pcld_use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
        num_kps: int = 8
            Number of keypoints to predict
        num_points: int 8192
            Number of sampled points from point clouds.
    """

    def __init__(
        self, num_classes, num_points=8192
    ):
        super(SEG, self).__init__()


        self.num_classes = num_classes
        self.cnn = ModifiedResnet()
        self.rgbd = DenseFusion(30000)
        self.conv2 = torch.nn.Conv1d(128, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.SEG_layer = (
            pt_utils.Seq(1024)
                .conv1d(512, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(num_classes, activation=None)
        )
        self.CtrOf_layer = (
            pt_utils.Seq(1792)
                .conv1d(1024, bn=True, activation=nn.ReLU())
                .conv1d(512, bn=True, activation=nn.ReLU())
                .conv1d(128, bn=True, activation=nn.ReLU())
                .conv1d(3, activation=None)
        )





    def forward(self,rgb, pcld, choose):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
            rgb: Variable(torch.cuda.FloatTensor)
                (B, C, H, W) tensor
            choose: Variable(torch.cuda.LongTensor)
                (B, 1, N) tensor
                indexs of choosen points(pixels).
        """
        out_rgb, rgb_seg = self.cnn(rgb)

        bs, di, _, _ = out_rgb.size()
        _, _, N, = choose.size()
        rgb_emb = out_rgb.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        rgb_emb = torch.gather(rgb_emb, 2, choose).contiguous()
        rgb_feature = self.conv2(rgb_emb)
        rgb_feature = self.conv3(rgb_feature)
        rgb_feature = self.conv4(rgb_feature)

        pcld_emb = pcld.transpose(1, 2).contiguous()
        rgbd_feature = self.rgbd(rgb_emb, pcld_emb)

        pred_rgbd_seg = self.SEG_layer(rgb_feature).transpose(1, 2).contiguous()
        pred_ctr_of = self.CtrOf_layer(rgbd_feature).view(
            bs, 1, 3, N
        )
        pred_ctr_of = pred_ctr_of.permute(0, 1, 3, 2).contiguous()

        return pred_rgbd_seg, pred_ctr_of




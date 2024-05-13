# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   arch_VHNet.py
@Time    :   2021/12/30 00:24:55
@Author  :   ZeyuXiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   None
"""

# import functools
# import torch
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F
# import math
# from skimage import morphology
# import numpy as np
# from dcn import ModulatedDeformConvPack, modulated_deform_conv

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from skimage import morphology
import numpy as np
import functools

class FeatureExtractor(nn.Module):
    def __init__(self, front_RBs, nf, ks):
        super(FeatureExtractor, self).__init__()
        self.front_RBs = front_RBs
        self.nf = nf
        self.ks = ks
        self.conv1_1 = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)
        self.conv1_2 = ResidualBlock_noBN(nf=self.nf)
        self.conv1_3 = ResidualBlock_noBN(nf=self.nf)
        self.conv2_1 = nn.Conv2d(self.nf, int(1.5*self.nf), 3, 2, 1, bias=True)
        self.conv2_2 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv2_3 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv3_1 = nn.Conv2d(int(1.5*self.nf), int(3*self.nf), 3, 2, 1, bias=True)
        self.conv3_2 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.conv3_3 = ResidualBlock_noBN(nf=int(3*self.nf))
        # self.ResASPP = ResASPP(int(3*self.nf))
    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv1_2(out1)
        out1 = self.conv1_3(out1)
        out2 = self.conv2_1(out1)
        out2 = self.conv2_2(out2)
        out2 = self.conv2_3(out2)
        out3 = self.conv3_1(out2)
        out3 = self.conv3_2(out3)
        out = self.conv3_3(out3)
        # out = self.ResASPP(out3)
        return out



class FeatureDecoder(nn.Module):
    def __init__(self, nf, ks):
        super(FeatureDecoder, self).__init__()
        self.nf = nf
        self.upconv3_i = nn.Conv2d(int(3*self.nf), int(3*self.nf), 3, 1, 1, bias=True)
        self.upconv3_1 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.upconv3_2 = ResidualBlock_noBN(nf=int(3*self.nf))
        # self.upconv3_3 = ResidualBlock_noBN(nf=int(3*self.nf))

        self.upconv2_u = nn.ConvTranspose2d(int(3*self.nf), int(1.5*self.nf), kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv2_i = nn.Conv2d(int(1.5*self.nf), int(1.5*self.nf), 3, 1, 1, bias=True)
        self.upconv2_1 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.upconv2_2 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        # self.upconv2_3 = ResidualBlock_noBN(nf=int(1.5*self.nf))

        self.upconv1_u = nn.ConvTranspose2d(int(1.5*self.nf), self.nf, kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv1_i = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.upconv1_1 = ResidualBlock_noBN(nf=self.nf)
        self.upconv1_2 = ResidualBlock_noBN(nf=self.nf)
        # self.upconv1_3 = ResidualBlock_noBN(nf=self.nf)

    def forward(self, x):
        out3 = self.upconv3_i(x)
        out3 = self.upconv3_1(out3)
        out3 = self.upconv3_2(out3)
        out2 = self.upconv2_u(out3)
        out2 = self.upconv2_i(out2)
        out2 = self.upconv2_1(out2)
        out2 = self.upconv2_2(out2)
        out1 = self.upconv1_u(out2)
        out1 = self.upconv1_i(out1)
        out1 = self.upconv1_1(out1)
        out = self.upconv1_2(out1)
        return out


class FeatureExtractor2(nn.Module):
    def __init__(self, front_RBs, nf, ks):
        super(FeatureExtractor2, self).__init__()
        self.front_RBs = front_RBs
        self.nf = nf
        self.ks = ks
        self.conv1_1 = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)
        self.conv1_2 = ResidualBlock_noBN(nf=self.nf)
        self.conv1_3 = ResidualBlock_noBN(nf=self.nf)
        self.conv1_4 = ResidualBlock_noBN(nf=self.nf)
        self.conv1_5 = ResidualBlock_noBN(nf=self.nf)
        self.conv2_1 = nn.Conv2d(self.nf, int(1.5*self.nf), 3, 2, 1, bias=True)
        self.conv2_2 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv2_3 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv2_4 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv2_5 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv3_1 = nn.Conv2d(int(1.5*self.nf), int(3*self.nf), 3, 2, 1, bias=True)
        self.conv3_2 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.conv3_3 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.conv3_4 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.conv3_5 = ResidualBlock_noBN(nf=int(3*self.nf))
        # self.ResASPP = ResASPP(int(3*self.nf))
    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv1_2(out1)
        out1 = self.conv1_3(out1)
        out1 = self.conv1_4(out1)
        out1 = self.conv1_5(out1)
        out2 = self.conv2_1(out1)
        out2 = self.conv2_2(out2)
        out2 = self.conv2_3(out2)
        out2 = self.conv2_4(out2)
        out2 = self.conv2_5(out2)
        out3 = self.conv3_1(out2)
        out3 = self.conv3_2(out3)
        out = self.conv3_3(out3)
        out = self.conv3_4(out3)
        out = self.conv3_5(out3)
        # out = self.ResASPP(out3)
        return out



class FeatureDecoder2(nn.Module):
    def __init__(self, nf, ks):
        super(FeatureDecoder2, self).__init__()
        self.nf = nf
        self.upconv3_i = nn.Conv2d(int(3*self.nf), int(3*self.nf), 3, 1, 1, bias=True)
        self.upconv3_1 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.upconv3_2 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.upconv3_3 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.upconv3_4 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.upconv3_5 = ResidualBlock_noBN(nf=int(3*self.nf))

        self.upconv2_u = nn.ConvTranspose2d(int(3*self.nf), int(1.5*self.nf), kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv2_i = nn.Conv2d(int(1.5*self.nf), int(1.5*self.nf), 3, 1, 1, bias=True)
        self.upconv2_1 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.upconv2_2 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.upconv2_3 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.upconv2_4 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.upconv2_5 = ResidualBlock_noBN(nf=int(1.5*self.nf))

        self.upconv1_u = nn.ConvTranspose2d(int(1.5*self.nf), self.nf, kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv1_i = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.upconv1_1 = ResidualBlock_noBN(nf=self.nf)
        self.upconv1_2 = ResidualBlock_noBN(nf=self.nf)
        self.upconv1_3 = ResidualBlock_noBN(nf=self.nf)
        self.upconv1_4 = ResidualBlock_noBN(nf=self.nf)
        self.upconv1_5 = ResidualBlock_noBN(nf=self.nf)

    def forward(self, x):
        out3 = self.upconv3_i(x)
        out3 = self.upconv3_1(out3)
        out3 = self.upconv3_2(out3)
        out3 = self.upconv3_3(out3)
        out3 = self.upconv3_4(out3)
        out3 = self.upconv3_5(out3)
        out2 = self.upconv2_u(out3)
        out2 = self.upconv2_i(out2)
        out2 = self.upconv2_1(out2)
        out2 = self.upconv2_2(out2)
        out2 = self.upconv2_3(out2)
        out2 = self.upconv2_4(out2)
        out2 = self.upconv2_5(out2)
        out1 = self.upconv1_u(out2)
        out1 = self.upconv1_i(out1)
        out1 = self.upconv1_1(out1)
        out = self.upconv1_2(out1)
        out = self.upconv1_3(out1)
        out = self.upconv1_4(out1)
        out = self.upconv1_5(out1)
        return out



class FeatureExtractor3(nn.Module):
    def __init__(self, front_RBs, nf, ks):
        super(FeatureExtractor3, self).__init__()
        self.front_RBs = front_RBs
        self.nf = nf
        self.ks = ks
        self.conv1_1 = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)
        self.conv1_2 = ResidualBlock_noBN(nf=self.nf)
        self.conv1_3 = CALayer(self.nf)
        self.conv1_4 = ResidualBlock_noBN(nf=self.nf)
        self.conv2_1 = nn.Conv2d(self.nf, int(1.5*self.nf), 3, 2, 1, bias=True)
        self.conv2_2 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv2_3 = CALayer(int(1.5*self.nf))
        self.conv2_4 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv3_1 = nn.Conv2d(int(1.5*self.nf), int(3*self.nf), 3, 2, 1, bias=True)
        self.conv3_2 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.conv3_3 = CALayer(int(3*self.nf))
        self.conv3_4 = ResidualBlock_noBN(nf=int(3*self.nf))
        # self.ResASPP = ResASPP(int(3*self.nf))
    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv1_2(out1)
        out1 = self.conv1_3(out1)
        out1 = self.conv1_4(out1)
        out2 = self.conv2_1(out1)
        out2 = self.conv2_2(out2)
        out2 = self.conv2_3(out2)
        out2 = self.conv2_4(out2)
        out3 = self.conv3_1(out2)
        out3 = self.conv3_2(out3)
        out = self.conv3_3(out3)
        out = self.conv3_4(out3)
        # out = self.ResASPP(out3)
        return out



class FeatureDecoder3(nn.Module):
    def __init__(self, nf, ks):
        super(FeatureDecoder3, self).__init__()
        self.nf = nf
        self.upconv3_i = nn.Conv2d(int(3*self.nf), int(3*self.nf), 3, 1, 1, bias=True)
        self.upconv3_1 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.upconv3_2 = CALayer(int(3*self.nf))
        self.upconv3_3 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.upconv3_4 = CALayer(int(3*self.nf))
        self.upconv3_5 = ResidualBlock_noBN(nf=int(3*self.nf))

        self.upconv2_u = nn.ConvTranspose2d(int(3*self.nf), int(1.5*self.nf), kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv2_i = nn.Conv2d(int(1.5*self.nf), int(1.5*self.nf), 3, 1, 1, bias=True)
        self.upconv2_1 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.upconv2_2 = CALayer(int(1.5*self.nf))
        self.upconv2_3 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.upconv2_4 = CALayer(int(1.5*self.nf))
        self.upconv2_5 = ResidualBlock_noBN(nf=int(1.5*self.nf))

        self.upconv1_u = nn.ConvTranspose2d(int(1.5*self.nf), self.nf, kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv1_i = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.upconv1_1 = ResidualBlock_noBN(nf=self.nf)
        self.upconv1_2 = CALayer(int(self.nf))
        self.upconv1_3 = ResidualBlock_noBN(nf=int(self.nf))
        self.upconv1_4 = CALayer(int(self.nf))
        self.upconv1_5 = ResidualBlock_noBN(nf=int(self.nf))

    def forward(self, x):
        out3 = self.upconv3_i(x)
        out3 = self.upconv3_1(out3)
        out3 = self.upconv3_2(out3)
        out3 = self.upconv3_3(out3)
        out3 = self.upconv3_4(out3)
        out3 = self.upconv3_5(out3)
        out2 = self.upconv2_u(out3)
        out2 = self.upconv2_i(out2)
        out2 = self.upconv2_1(out2)
        out2 = self.upconv2_2(out2)
        out2 = self.upconv2_3(out2)
        out2 = self.upconv2_4(out2)
        out2 = self.upconv2_5(out2)
        out1 = self.upconv1_u(out2)
        out1 = self.upconv1_i(out1)
        out1 = self.upconv1_1(out1)
        out = self.upconv1_2(out1)
        out = self.upconv1_3(out1)
        out = self.upconv1_4(out1)
        out = self.upconv1_5(out1)
        return out


class FeatureExtractor4(nn.Module):
    def __init__(self, front_RBs, nf, ks):
        super(FeatureExtractor4, self).__init__()
        self.front_RBs = front_RBs
        self.nf = nf
        self.ks = ks
        self.conv1_1 = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)
        self.conv1_2 = ResidualBlock_noBN(nf=self.nf)
        self.conv1_3 = CALayer(self.nf)
        self.conv1_4 = ResidualBlock_noBN(nf=self.nf)
        self.conv2_1 = nn.Conv2d(self.nf, int(1.5*self.nf), 3, 2, 1, bias=True)
        self.conv2_2 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv2_3 = CALayer(int(1.5*self.nf))
        self.conv2_4 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.conv3_1 = nn.Conv2d(int(1.5*self.nf), int(3*self.nf), 3, 2, 1, bias=True)
        self.conv3_2 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.conv3_3 = CALayer(int(3*self.nf))
        self.conv3_4 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.ResASPP = ResASPP(int(3*self.nf))

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv1_2(out1)
        out1 = self.conv1_3(out1)
        out1 = self.conv1_4(out1)
        out2 = self.conv2_1(out1)
        out2 = self.conv2_2(out2)
        out2 = self.conv2_3(out2)
        out2 = self.conv2_4(out2)
        out3 = self.conv3_1(out2)
        out3 = self.conv3_2(out3)
        out = self.conv3_3(out3)
        out = self.conv3_4(out3)
        out = self.ResASPP(out3)
        return out

class FeatureDecoder4(nn.Module):
    def __init__(self, nf, ks):
        super(FeatureDecoder4, self).__init__()
        self.nf = nf
        self.upconv3_i = nn.Conv2d(int(3*self.nf), int(3*self.nf), 3, 1, 1, bias=True)
        self.upconv3_1 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.upconv3_2 = CALayer(int(3*self.nf))
        self.upconv3_3 = ResidualBlock_noBN(nf=int(3*self.nf))
        self.upconv3_4 = CALayer(int(3*self.nf))
        self.upconv3_5 = ResidualBlock_noBN(nf=int(3*self.nf))

        self.upconv2_u = nn.ConvTranspose2d(int(3*self.nf), int(1.5*self.nf), kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv2_i = nn.Conv2d(int(1.5*self.nf), int(1.5*self.nf), 3, 1, 1, bias=True)
        self.upconv2_1 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.upconv2_2 = CALayer(int(1.5*self.nf))
        self.upconv2_3 = ResidualBlock_noBN(nf=int(1.5*self.nf))
        self.upconv2_4 = CALayer(int(1.5*self.nf))
        self.upconv2_5 = ResidualBlock_noBN(nf=int(1.5*self.nf))

        self.upconv1_u = nn.ConvTranspose2d(int(1.5*self.nf), self.nf, kernel_size=4, stride=2, padding=1, bias=True)
        self.upconv1_i = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.upconv1_1 = ResidualBlock_noBN(nf=self.nf)
        self.upconv1_2 = CALayer(int(self.nf))
        self.upconv1_3 = ResidualBlock_noBN(nf=int(self.nf))
        self.upconv1_4 = CALayer(int(self.nf))
        self.upconv1_5 = ResidualBlock_noBN(nf=int(self.nf))

    def forward(self, x):
        out3 = self.upconv3_i(x)
        out3 = self.upconv3_1(out3)
        out3 = self.upconv3_2(out3)
        out3 = self.upconv3_3(out3)
        out3 = self.upconv3_4(out3)
        out3 = self.upconv3_5(out3)
        out2 = self.upconv2_u(out3)
        out2 = self.upconv2_i(out2)
        out2 = self.upconv2_1(out2)
        out2 = self.upconv2_2(out2)
        out2 = self.upconv2_3(out2)
        out2 = self.upconv2_4(out2)
        out2 = self.upconv2_5(out2)
        out1 = self.upconv1_u(out2)
        out1 = self.upconv1_i(out1)
        out1 = self.upconv1_1(out1)
        out = self.upconv1_2(out1)
        out = self.upconv1_3(out1)
        out = self.upconv1_4(out1)
        out = self.upconv1_5(out1)
        return out




class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel,channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        # self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2,
        #                                       dilation=2, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4,
                                              dilation=4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        # buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


class DRAM(nn.Module):
    def __init__(self, nf):
        super(DRAM, self).__init__()
        self.conv_down_a = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_a = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_down_b = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_up_b = nn.ConvTranspose2d(nf, nf, 3, 2, 1, 1, bias=True)
        self.conv_cat = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, lr, ref):
        res_a = self.act(self.conv_down_a(ref)) - lr
        out_a = self.act(self.conv_up_a(res_a)) + ref

        res_b = lr - self.act(self.conv_down_b(ref))
        out_b = self.act(self.conv_up_b(res_b + lr))

        out = self.act(self.conv_cat(torch.cat([out_a, out_b], dim=1)))

        return out



class GlobalAttentionBlock(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(GlobalAttentionBlock, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//32, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//32, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class GlobalAttentionBlock2(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(GlobalAttentionBlock2, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
        self.conv1 = nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def conv_extractor(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )

def upconv_extractor(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicAttentionBlockv1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicAttentionBlockv1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, 1, padding=0, bias=True)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, inplanes, 1, padding=0, bias=True)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.ca = CALayer(planes)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        out += residual
        out = self.relu(out)

        return out




def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat



class Upsampler(nn.Sequential):  # 上采样/放大器
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        # scale:放大倍数
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):  # 2^n,循环n次
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                # PixelShuffle,(1,c*4,h,w)----->(1,c,h*2,w*2) 把通道的像素迁移到size上
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


if __name__ == "__main__":
    x1 = torch.rand(1, 64, 64,64).cuda()
    print((x1/2).shape)
    model = ResWTBlock(embed_ch=64).cuda()
    a= model(x1)
    print(a.shape)
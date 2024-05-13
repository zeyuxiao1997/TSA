# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   fusionModule.py
@Time    :   2021/12/28 21:01:56
@Author  :   ZeyuXiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   将三个group中output的信息做聚合的模块 fusion模块
"""

# import functools
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import arch_VHNet as arch_VHNet

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.arch_VHNet as arch_VHNet

class FusionModule1(nn.Module):
    """
    TSA模块去掉后面的pyramid部分，只有2D的部分
    """
    def __init__(self, nf=64, groups=3, center=1):
        super(FusionModule1, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.temAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.temAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(groups * nf, nf, 1, 1, bias=True)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv3d = nn.Conv3d(groups,groups,3,1,1,bias=True)####################new
        self.conv3d_2d =  nn.Conv2d(groups*nf,nf,3,1,1,bias=True)

    def forward(self, outGroupAll):
        x = outGroupAll
        B, N, C, H, W = outGroupAll.size()  # N video frames
        #### temporal attention
        emb_ref = self.temAtt_2(outGroupAll[:, self.center, :, :, :].clone())
        emb = self.temAtt_1(outGroupAll.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        outGroupAll = outGroupAll.view(B, -1, H, W) * cor_prob

        #### fusion
        fea_3d = self.conv3d_2d(self.conv3d(x).view(B,-1,H,W))##########################################################new
        fea = self.lrelu(self.fea_fusion(outGroupAll))+fea_3d

        return fea


class FusionModule2(nn.Module):
    """
    TSA模块去掉后面的pyramid部分，只有2D的部分
    这个直接是TSA模块，去掉了模型1的3D卷积部分
    """
    def __init__(self, nf=64, groups=3, center=1):
        super(FusionModule2, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.temAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.temAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(groups * nf, nf, 1, 1, bias=True)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv3d = nn.Conv3d(groups,groups,3,1,1,bias=True)####################new
        self.conv3d_2d =  nn.Conv2d(groups*nf,nf,3,1,1,bias=True)

    def forward(self, outGroupAll):
        x = outGroupAll
        B, N, C, H, W = outGroupAll.size()  # N video frames
        #### temporal attention
        emb_ref = self.temAtt_2(outGroupAll[:, self.center, :, :, :].clone())
        emb = self.temAtt_1(outGroupAll.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        outGroupAll = outGroupAll.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.fea_fusion(outGroupAll)

        return fea

class FusionModule3(nn.Module):
    """
    TSA模块去掉后面的pyramid部分，只有2D的部分
    原来网络里面的3D conv变成好几个3D residual conv（从input直接揉） 4个
    """
    def __init__(self, nf=64, groups=3, center=1):
        super(FusionModule3, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.temAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.temAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(groups * nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv3d = nn.Conv3d(groups,groups,3,1,1,bias=True)
        self.conv3d_1 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_2 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_3 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_4 = arch_VHNet.ResBlock_3d(nf)
        # self.conv3d_5 = arch_VHNet.ResBlock_3d(nf)
        # self.conv3d_6 = arch_VHNet.ResBlock_3d(nf)
        # self.conv3d_7 = arch_VHNet.ResBlock_3d(nf)
        # self.conv3d_8 = arch_VHNet.ResBlock_3d(nf)

        ####################new
        self.conv3d_2d =  nn.Conv2d(groups*nf,nf,3,1,1,bias=True)

    def forward(self, outGroupAll):
        x = outGroupAll
        B, N, C, H, W = outGroupAll.size()  # N video frames
        #### temporal attention
        emb_ref = self.temAtt_2(outGroupAll[:, self.center, :, :, :].clone())
        emb = self.temAtt_1(outGroupAll.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        outGroupAll = outGroupAll.view(B, -1, H, W) * cor_prob

        #### fusion
        fea_3d = self.conv3d(x)
        fea_3d = self.conv3d_1(fea_3d.view(B, -1, N, H, W))
        fea_3d = self.conv3d_2(fea_3d)
        fea_3d = self.conv3d_3(fea_3d)
        fea_3d = self.conv3d_4(fea_3d)
        # fea_3d = self.conv3d_5(fea_3d)
        # fea_3d = self.conv3d_6(fea_3d)
        # fea_3d = self.conv3d_7(fea_3d)
        # fea_3d = self.conv3d_8(fea_3d)
        fea_3d = self.conv3d_2d(fea_3d.view(B,-1,H,W))##########################################################new
        fea = self.lrelu(self.fea_fusion(outGroupAll))+fea_3d

        return fea


class FusionModule4(nn.Module):
    """
    TSA模块去掉后面的pyramid部分，只有2D的部分
    原来网络里面的3D conv变成好几个3D residual conv（从input直接揉） 10个
    """
    def __init__(self, nf=64, groups=3, center=1):
        super(FusionModule4, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.temAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.temAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(groups * nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv3d = nn.Conv3d(groups,groups,3,1,1,bias=True)
        self.conv3d_1 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_2 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_3 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_4 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_5 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_6 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_7 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_8 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_9 = arch_VHNet.ResBlock_3d(nf)
        self.conv3d_10 = arch_VHNet.ResBlock_3d(nf)
        ####################new
        self.conv3d_2d =  nn.Conv2d(groups*nf,nf,3,1,1,bias=True)

    def forward(self, outGroupAll):
        x = outGroupAll
        B, N, C, H, W = outGroupAll.size()  # N video frames
        #### temporal attention
        emb_ref = self.temAtt_2(outGroupAll[:, self.center, :, :, :].clone())
        emb = self.temAtt_1(outGroupAll.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        outGroupAll = outGroupAll.view(B, -1, H, W) * cor_prob

        #### fusion
        fea_3d = self.conv3d(x)
        fea_3d = self.conv3d_1(fea_3d.view(B, -1, N, H, W))
        fea_3d = self.conv3d_2(fea_3d)
        fea_3d = self.conv3d_3(fea_3d)
        fea_3d = self.conv3d_4(fea_3d)
        fea_3d = self.conv3d_5(fea_3d)
        fea_3d = self.conv3d_6(fea_3d)
        fea_3d = self.conv3d_7(fea_3d)
        fea_3d = self.conv3d_8(fea_3d)
        fea_3d = self.conv3d_9(fea_3d)
        fea_3d = self.conv3d_10(fea_3d)
        fea_3d = self.conv3d_2d(fea_3d.view(B,-1,H,W))##########################################################new
        fea = self.lrelu(self.fea_fusion(outGroupAll))+fea_3d

        return fea


class FusionModule5(nn.Module):
    """
    TSA模块去掉后面的pyramid部分，只有2D的部分
    后面加上spatial attention
    """
    def __init__(self, nf=64, groups=3, center=1):
        super(FusionModule5, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.temAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.temAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(groups * nf, nf, 1, 1, bias=True)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv3d = nn.Conv3d(groups,groups,3,1,1,bias=True)####################new
        self.conv3d_2d =  nn.Conv2d(groups*nf,nf,3,1,1,bias=True)

        self.conv_1 = arch_VHNet.GlobalAttentionBlock(nf)
        self.conv_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_3 = arch_VHNet.GlobalAttentionBlock(nf)
        self.conv_4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_5 = arch_VHNet.GlobalAttentionBlock(nf)
        self.conv_6 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_7 = arch_VHNet.GlobalAttentionBlock(nf)
        self.conv_8 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, outGroupAll):
        x = outGroupAll
        B, N, C, H, W = outGroupAll.size()  # N video frames
        #### temporal attention
        emb_ref = self.temAtt_2(outGroupAll[:, self.center, :, :, :].clone())
        emb = self.temAtt_1(outGroupAll.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        outGroupAll = outGroupAll.view(B, -1, H, W) * cor_prob

        #### fusion
        fea_3d = self.conv3d_2d(self.conv3d(x).view(B,-1,H,W))##########################################################new
        fea = self.lrelu(self.fea_fusion(outGroupAll))+fea_3d

        fea = self.conv_1(fea)
        fea = self.conv_2(fea)
        fea = self.conv_3(fea)
        fea = self.conv_4(fea)
        fea = self.conv_5(fea)
        fea = self.conv_6(fea)
        fea = self.conv_7(fea)
        fea = self.conv_8(fea)
        return fea

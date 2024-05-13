# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   intraModule.py
@Time    :   2021/12/26 11:44:19
@Author  :   ZeyuXiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   每个group里面的子模块
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


class IntraModule1(nn.Module):
    """
    基于AttAdaIN的魔改版本，
    attention map由all和center生成，后面用BG和FG的region揉（一模一样）
    transfer之后的FG feature和BG feature做sum，然后用CBAM的模块
    """
    def __init__(self, nf):
        self.nf = nf
        super(IntraModule1, self).__init__()
        self.conv_group_1 = nn.Conv2d(3*self.nf, 3*self.nf, 1, 1, 0, bias=True)
        self.calayer_group = arch_VHNet.CALayer(3*nf)
        self.conv_group_2 = nn.Conv2d(3*self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv_group_3 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.conv_center_1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.calayer_center = arch_VHNet.CALayer(self.nf)
        self.conv_center_2 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.f = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.g = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.h = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.sm = nn.Softmax(dim=-1)

        self.BasicAttentionBlockv1 = arch_VHNet.BasicAttentionBlockv1(self.nf, self.nf)

        
    def forward(self, group, FG, BG):  # B * C * H * W
        Bg,Ng,Cg,Hg,Wg = group.shape
        center = group[:,1,:,:,:]
        FG = FG[:,1,:,:,:]
        BG = BG[:,1,:,:,:]
        group = group.view(Bg,-1,Hg,Wg)
        fea_group = self.conv_group_1(group)
        fea_group = self.calayer_group(fea_group)
        fea_group = self.conv_group_2(fea_group)
        fea_group = self.conv_group_3(fea_group)

        fea_center = self.conv_center_1(center)
        fea_center = self.calayer_center(fea_center)
        fea_center = self.conv_center_2(fea_center)

        F = self.f(fea_group)
        G = self.g(fea_center)
        H = self.h(BG)

        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        style_BG_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_BG_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_BG_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        transfer_FG = std * arch_VHNet.mean_variance_norm(FG) + mean

        out = self.BasicAttentionBlockv1(transfer_FG+BG)
        return out


class IntraModule2(nn.Module):
    """
    基于AttAdaIN的魔改版本，
    attention map由all和center生成，后面用BG和FG的region揉（一模一样）
    transfer之后的FG feature和BG feature做concat，然后用CBAM的模块，然后用conv1x1
    """
    def __init__(self, nf):
        self.nf = nf
        super(IntraModule2, self).__init__()
        self.conv_group_1 = nn.Conv2d(3*self.nf, 3*self.nf, 1, 1, 0, bias=True)
        self.calayer_group = arch_VHNet.CALayer(3*nf)
        self.conv_group_2 = nn.Conv2d(3*self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv_group_3 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.conv_center_1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.calayer_center = arch_VHNet.CALayer(self.nf)
        self.conv_center_2 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.f = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.g = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.h = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.sm = nn.Softmax(dim=-1)

        self.BasicAttentionBlockv1 = arch_VHNet.BasicAttentionBlockv1(2*self.nf, 2*self.nf)
        self.conv_last = nn.Conv2d(2*self.nf, self.nf, 1, 1, 0, bias=True)

        
    def forward(self, group, FG, BG):  # B * C * H * W
        Bg, Ng, Cg, Hg, Wg = group.size()
        center = group[:,1,:,:,:]
        FG = FG[:,1,:,:,:]
        BG = BG[:,1,:,:,:]
        group = group.view(Bg,-1,Hg,Wg)
        fea_group = self.conv_group_1(group)
        fea_group = self.calayer_group(fea_group)
        fea_group = self.conv_group_2(fea_group)
        fea_group = self.conv_group_3(fea_group)

        fea_center = self.conv_center_1(center)
        fea_center = self.calayer_center(fea_center)
        fea_center = self.conv_center_2(fea_center)

        F = self.f(fea_group)
        G = self.g(fea_center)
        H = self.h(BG)

        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        style_BG_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_BG_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_BG_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        transfer_FG = std * arch_VHNet.mean_variance_norm(FG) + mean

        fea = torch.cat((transfer_FG,BG),dim=1)
        print('llll',fea.shape)
        out = self.BasicAttentionBlockv1(fea)
        out = self.conv_last(out)
        return out

class IntraModule3(nn.Module):
    """
    基于AttAdaIN的魔改版本，
    attention map由all生成（分两支生成），后面用BG和FG的region揉（一模一样）
    transfer之后的FG feature和BG feature做sum，然后用CBAM的模块
    """
    def __init__(self, nf):
        self.nf = nf
        super(IntraModule3, self).__init__()
        self.conv_group_1 = nn.Conv2d(3*self.nf, 3*self.nf, 1, 1, 0, bias=True)
        self.calayer_group = arch_VHNet.CALayer(3*nf)
        self.conv_group_2 = nn.Conv2d(3*self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv_group_3 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.conv_center_1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.calayer_center = arch_VHNet.CALayer(self.nf)
        self.conv_center_2 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.f = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.g = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.h = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.sm = nn.Softmax(dim=-1)

        self.BasicAttentionBlockv1 = arch_VHNet.BasicAttentionBlockv1(self.nf, self.nf)

        
    def forward(self, group, FG, BG):  # B * C * H * W
        Bg,Ng,Cg,Hg,Wg = group.shape
        center = group[:,1,:,:,:]
        FG = FG[:,1,:,:,:]
        BG = BG[:,1,:,:,:]
        group = group.view(Bg,-1,Hg,Wg)
        fea_group = self.conv_group_1(group)
        fea_group = self.calayer_group(fea_group)
        fea_group = self.conv_group_2(fea_group)
        fea_group = self.conv_group_3(fea_group)

        # fea_center = self.conv_center_1(center)
        # fea_center = self.calayer_center(fea_center)
        # fea_center = self.conv_center_2(fea_center)

        F = self.f(fea_group)
        G = self.g(fea_group)
        H = self.h(BG)

        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        style_BG_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_BG_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_BG_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        transfer_FG = std * arch_VHNet.mean_variance_norm(FG) + mean

        out = self.BasicAttentionBlockv1(transfer_FG+BG)
        return out



class IntraModule4(nn.Module):
    """
    基于jiajiaya的MASASR的魔改版本，基于SAM做魔改
    原本的LR就是background，原本的ref就是不和谐的前景
    【1-5方案中，4的方案最好，后面6-9是基于4的模板版本】
    """
    def __init__(self, nf):
        self.nf = nf
        super(IntraModule4, self).__init__()
        self.conv_group_1 = nn.Conv2d(3*self.nf, 3*self.nf, 1, 1, 0, bias=True)
        self.calayer_group = arch_VHNet.CALayer(3*nf)
        self.conv_group_2 = nn.Conv2d(3*self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv_group_3 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.conv_center_1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.calayer_center = arch_VHNet.CALayer(self.nf)
        self.conv_center_2 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)


        self.norm_layer = nn.InstanceNorm2d(self.nf, affine=False)


        self.conv_shared = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=True),
                                            nn.ReLU(inplace=True))
        self.conv_gamma = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        # initialization
        self.conv_gamma.weight.data.zero_()
        self.conv_beta.weight.data.zero_()
        self.conv_gamma.bias.data.zero_()
        self.conv_beta.bias.data.zero_()


    def forward(self, group, FG, BG):  # B * C * H * W
        Bg,Ng,Cg,Hg,Wg = BG.shape
        center = group[:,1,:,:,:]
        FG = FG[:,1,:,:,:]
        # BG = BG[:,1,:,:,:]
        FG_normed = self.norm_layer(FG)

        BG = BG.view(Bg,-1,Hg,Wg)
        fea_BG = self.conv_group_1(BG)
        fea_BG = self.calayer_group(fea_BG)
        fea_BG = self.conv_group_2(fea_BG)
        fea_BG = self.conv_group_3(fea_BG)

        fea_FG = self.conv_center_1(FG)
        fea_FG = self.calayer_center(fea_FG)
        fea_FG = self.conv_center_2(fea_FG)

        style = self.conv_shared(torch.cat([fea_BG, fea_FG], dim=1))
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)

        b, c, h, w = fea_BG.size()
        fea_FG = fea_FG.view(b, c, h * w)
        fea_FG_mean = torch.mean(fea_FG, dim=-1, keepdim=True).unsqueeze(3)
        fea_FG_std = torch.std(fea_FG, dim=-1, keepdim=True).unsqueeze(3)

        gamma = gamma + fea_FG_std
        beta = beta + fea_FG_mean


        out = FG_normed * gamma + beta

        return out



class IntraModule5(nn.Module):
    """
    基于jiajiaya的MASASR的魔改版本，基于SAM做魔改
    原本的LR就是background，原本的ref就是不和谐的前景
    接4，FG和FG_norm的做fusion，补充归一化丢失信息（类似于黄杰的操作）
    """
    def __init__(self, nf):
        self.nf = nf
        super(IntraModule5, self).__init__()
        self.conv_group_1 = nn.Conv2d(3*self.nf, 3*self.nf, 1, 1, 0, bias=True)
        self.calayer_group = arch_VHNet.CALayer(3*nf)
        self.conv_group_2 = nn.Conv2d(3*self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv_group_3 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.conv_center_1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.calayer_center = arch_VHNet.CALayer(self.nf)
        self.conv_center_2 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)


        self.norm_layer = nn.InstanceNorm2d(self.nf, affine=False)


        self.conv_shared = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=True),
                                            nn.ReLU(inplace=True))
        self.conv_gamma = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        # initialization
        self.conv_gamma.weight.data.zero_()
        self.conv_beta.weight.data.zero_()
        self.conv_gamma.bias.data.zero_()
        self.conv_beta.bias.data.zero_()

        # 补充信息
        self.conv_last_1 = nn.Conv2d(2*self.nf, 2*self.nf, 1, 1, 0, bias=True)
        self.calayer_last = arch_VHNet.CALayer(2*nf)
        self.conv_last_2 = nn.Conv2d(2*self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv_last_3 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)



    def forward(self, group, FG, BG):  # B * C * H * W
        Bg,Ng,Cg,Hg,Wg = BG.shape
        center = group[:,1,:,:,:]
        FG = FG[:,1,:,:,:]
        # BG = BG[:,1,:,:,:]
        FG_normed = self.norm_layer(FG)

        BG = BG.view(Bg,-1,Hg,Wg)
        fea_BG = self.conv_group_1(BG)
        fea_BG = self.calayer_group(fea_BG)
        fea_BG = self.conv_group_2(fea_BG)
        fea_BG = self.conv_group_3(fea_BG)

        fea_FG = self.conv_center_1(FG)
        fea_FG = self.calayer_center(fea_FG)
        fea_FG = self.conv_center_2(fea_FG)

        style = self.conv_shared(torch.cat([fea_BG, fea_FG], dim=1))
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)

        b, c, h, w = fea_BG.size()
        fea_FG = fea_FG.view(b, c, h * w)
        fea_FG_mean = torch.mean(fea_FG, dim=-1, keepdim=True).unsqueeze(3)
        fea_FG_std = torch.std(fea_FG, dim=-1, keepdim=True).unsqueeze(3)

        gamma = gamma + fea_FG_std
        beta = beta + fea_FG_mean

        out = FG_normed * gamma + beta

        out = self.conv_last_1(torch.cat([out, FG_normed], dim=1))
        out = self.calayer_last(out)
        out = self.conv_last_2(out)
        out = self.conv_last_3(out)
        return out


class IntraModule6(nn.Module):
    """
    基于4方案的魔改版本
    基于jiajiaya的MASASR的魔改版本，基于SAM做魔改
    原本的LR就是background，原本的ref就是不和谐的前景
    【1-5方案中，4的方案最好，后面6-9是基于4的魔改版本】
    把方案4中的conv变成3x3的大小，原来是1x1
    """
    def __init__(self, nf):
        self.nf = nf
        super(IntraModule6, self).__init__()
        self.conv_group_1 = nn.Conv2d(3*self.nf, 3*self.nf, 3, 1, 1, bias=True)
        self.calayer_group = arch_VHNet.CALayer(3*nf)
        self.conv_group_2 = nn.Conv2d(3*self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_group_3 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        self.conv_center_1 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.calayer_center = arch_VHNet.CALayer(self.nf)
        self.conv_center_2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        self.norm_layer = nn.InstanceNorm2d(self.nf, affine=False)

        self.conv_shared = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=True),
                                            nn.ReLU(inplace=True))
        self.conv_gamma = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        # initialization
        self.conv_gamma.weight.data.zero_()
        self.conv_beta.weight.data.zero_()
        self.conv_gamma.bias.data.zero_()
        self.conv_beta.bias.data.zero_()

        ## 细节迁移分支
        # self.conv_detail_1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)



    def forward(self, group, FG, BG):  # B * C * H * W
        Bg,Ng,Cg,Hg,Wg = BG.shape
        center = group[:,1,:,:,:]
        FG = FG[:,1,:,:,:]
        # BG = BG[:,1,:,:,:]
        FG_normed = self.norm_layer(FG)

        BG = BG.view(Bg,-1,Hg,Wg)
        fea_BG = self.conv_group_1(BG)
        fea_BG = self.calayer_group(fea_BG)
        fea_BG = self.conv_group_2(fea_BG)
        fea_BG = self.conv_group_3(fea_BG)

        fea_FG = self.conv_center_1(FG)
        fea_FG = self.calayer_center(fea_FG)
        fea_FG = self.conv_center_2(fea_FG)

        style = self.conv_shared(torch.cat([fea_BG, fea_FG], dim=1))
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)

        b, c, h, w = fea_BG.size()
        fea_FG = fea_FG.view(b, c, h * w)
        fea_FG_mean = torch.mean(fea_FG, dim=-1, keepdim=True).unsqueeze(3)
        fea_FG_std = torch.std(fea_FG, dim=-1, keepdim=True).unsqueeze(3)

        gamma = gamma + fea_FG_std
        beta = beta + fea_FG_mean

        out = FG_normed * gamma + beta

        return out


class IntraModule7(nn.Module):
    """
    基于4方案的魔改版本
    基于jiajiaya的MASASR的魔改版本，基于SAM做魔改
    原本的LR就是background，原本的ref就是不和谐的前景
    【1-5方案中，4的方案最好，后面6-9是基于4的魔改版本】
    迁移之后的FG特征再和BG特征做聚合,用dual-attention里面的面揉
    """
    def __init__(self, nf):
        self.nf = nf
        super(IntraModule7, self).__init__()
        self.conv_group_1 = nn.Conv2d(3*self.nf, 3*self.nf, 1, 1, 0, bias=True)
        self.calayer_group = arch_VHNet.CALayer(3*nf)
        self.conv_group_2 = nn.Conv2d(3*self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv_group_3 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.conv_center_1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.calayer_center = arch_VHNet.CALayer(self.nf)
        self.conv_center_2 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)


        self.norm_layer = nn.InstanceNorm2d(self.nf, affine=False)

        self.conv_shared = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=True),
                                            nn.ReLU(inplace=True))
        self.conv_gamma = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        # initialization
        self.conv_gamma.weight.data.zero_()
        self.conv_beta.weight.data.zero_()
        self.conv_gamma.bias.data.zero_()
        self.conv_beta.bias.data.zero_()

        ## 细节迁移分支
        self.conv_detail_1 = nn.Conv2d(2*self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_detail_2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_detail_3 = arch_VHNet.GlobalAttentionBlock(self.nf)
        self.conv_detail_4 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)



    def forward(self, group, FG, BG, mask_center):  # B * C * H * W
        Bg,Ng,Cg,Hg,Wg = BG.shape
        center = group[:,1,:,:,:]
        FG = FG[:,1,:,:,:]
        # BG = BG[:,1,:,:,:]
        FG_normed = self.norm_layer(FG)

        BG = BG.view(Bg,-1,Hg,Wg)
        fea_BG = self.conv_group_1(BG)
        fea_BG = self.calayer_group(fea_BG)
        fea_BG = self.conv_group_2(fea_BG)
        fea_BG = self.conv_group_3(fea_BG)

        fea_FG = self.conv_center_1(FG)
        fea_FG = self.calayer_center(fea_FG)
        fea_FG = self.conv_center_2(fea_FG)

        style = self.conv_shared(torch.cat([fea_BG, fea_FG], dim=1))
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)

        b, c, h, w = fea_BG.size()
        fea_FG = fea_FG.view(b, c, h * w)
        fea_FG_mean = torch.mean(fea_FG, dim=-1, keepdim=True).unsqueeze(3)
        fea_FG_std = torch.std(fea_FG, dim=-1, keepdim=True).unsqueeze(3)

        gamma = gamma + fea_FG_std
        beta = beta + fea_FG_mean

        out = FG_normed * gamma + beta

        ####################################################################
        # 加一个non-local
        mask_center = F.interpolate(mask_center, scale_factor=1/4, mode='bilinear', align_corners=False)
        # print('mask.shape',mask_center.shape)
        detail = self.conv_detail_1(torch.cat([out, fea_BG], dim=1))
        # print(detail.shape)
        detail = self.conv_detail_2(detail)
        detail = self.conv_detail_3(detail)
        detail = detail * mask_center + out
        detail = self.conv_detail_4(detail)
        return out


class IntraModule8(nn.Module):
    """
    基于4方案的魔改版本
    基于jiajiaya的MASASR的魔改版本，基于SAM做魔改
    原本的LR就是background，原本的ref就是不和谐的前景
    【1-5方案中，4的方案最好，后面6-9是基于4的魔改版本】
    迁移之后的FG特征再和BG特征做聚合,用dual-attention里面的面揉，相比于模型7少了out的res
    """
    def __init__(self, nf):
        self.nf = nf
        super(IntraModule8, self).__init__()
        self.conv_group_1 = nn.Conv2d(3*self.nf, 3*self.nf, 1, 1, 0, bias=True)
        self.calayer_group = arch_VHNet.CALayer(3*nf)
        self.conv_group_2 = nn.Conv2d(3*self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv_group_3 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.conv_center_1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.calayer_center = arch_VHNet.CALayer(self.nf)
        self.conv_center_2 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)


        self.norm_layer = nn.InstanceNorm2d(self.nf, affine=False)

        self.conv_shared = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=True),
                                            nn.ReLU(inplace=True))
        self.conv_gamma = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        # initialization
        self.conv_gamma.weight.data.zero_()
        self.conv_beta.weight.data.zero_()
        self.conv_gamma.bias.data.zero_()
        self.conv_beta.bias.data.zero_()

        ## 细节迁移分支
        self.conv_detail_1 = nn.Conv2d(2*self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_detail_2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_detail_3 = arch_VHNet.GlobalAttentionBlock(self.nf)
        self.conv_detail_4 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

    def forward(self, group, FG, BG, mask_center):  # B * C * H * W
        Bg,Ng,Cg,Hg,Wg = BG.shape
        center = group[:,1,:,:,:]
        FG = FG[:,1,:,:,:]
        # BG = BG[:,1,:,:,:]
        FG_normed = self.norm_layer(FG)

        BG = BG.view(Bg,-1,Hg,Wg)
        fea_BG = self.conv_group_1(BG)
        fea_BG = self.calayer_group(fea_BG)
        fea_BG = self.conv_group_2(fea_BG)
        fea_BG = self.conv_group_3(fea_BG)

        fea_FG = self.conv_center_1(FG)
        fea_FG = self.calayer_center(fea_FG)
        fea_FG = self.conv_center_2(fea_FG)

        style = self.conv_shared(torch.cat([fea_BG, fea_FG], dim=1))
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)

        b, c, h, w = fea_BG.size()
        fea_FG = fea_FG.view(b, c, h * w)
        fea_FG_mean = torch.mean(fea_FG, dim=-1, keepdim=True).unsqueeze(3)
        fea_FG_std = torch.std(fea_FG, dim=-1, keepdim=True).unsqueeze(3)

        gamma = gamma + fea_FG_std
        beta = beta + fea_FG_mean

        out = FG_normed * gamma + beta

        ####################################################################
        # 加一个non-local
        mask_center = F.interpolate(mask_center, scale_factor=1/4, mode='bilinear', align_corners=False)
        # print('mask.shape',mask_center.shape)
        detail = self.conv_detail_1(torch.cat([out, fea_BG], dim=1))
        # print(detail.shape)
        detail = self.conv_detail_2(detail)
        detail = self.conv_detail_3(detail)
        detail = detail * mask_center
        detail = self.conv_detail_4(detail)
        return out


class IntraModule9(nn.Module):
    """
    基于4方案的魔改版本
    基于jiajiaya的MASASR的魔改版本，基于SAM做魔改
    原本的LR就是background，原本的ref就是不和谐的前景
    【1-5方案中，4的方案最好，后面6-9是基于4的魔改版本】
    迁移之后的FG特征再和BG特征做聚合,用dual-attention里面的面揉，相比于模型7改了spatial attention的matrix的channel数，从//32变成//4
    """
    def __init__(self, nf):
        self.nf = nf
        super(IntraModule9, self).__init__()
        self.conv_group_1 = nn.Conv2d(3*self.nf, 3*self.nf, 1, 1, 0, bias=True)
        self.calayer_group = arch_VHNet.CALayer(3*nf)
        self.conv_group_2 = nn.Conv2d(3*self.nf, self.nf, 1, 1, 0, bias=True)
        self.conv_group_3 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)

        self.conv_center_1 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)
        self.calayer_center = arch_VHNet.CALayer(self.nf)
        self.conv_center_2 = nn.Conv2d(self.nf, self.nf, 1, 1, 0, bias=True)


        self.norm_layer = nn.InstanceNorm2d(self.nf, affine=False)

        self.conv_shared = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=True),
                                            nn.ReLU(inplace=True))
        self.conv_gamma = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_beta = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        # initialization
        self.conv_gamma.weight.data.zero_()
        self.conv_beta.weight.data.zero_()
        self.conv_gamma.bias.data.zero_()
        self.conv_beta.bias.data.zero_()

        ## 细节迁移分支
        self.conv_detail_1 = nn.Conv2d(2*self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_detail_2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.conv_detail_3 = arch_VHNet.GlobalAttentionBlock2(self.nf)
        self.conv_detail_4 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

    def forward(self, group, FG, BG, mask_center):  # B * C * H * W
        Bg,Ng,Cg,Hg,Wg = BG.shape
        center = group[:,1,:,:,:]
        FG = FG[:,1,:,:,:]
        # BG = BG[:,1,:,:,:]
        FG_normed = self.norm_layer(FG)

        BG = BG.view(Bg,-1,Hg,Wg)
        fea_BG = self.conv_group_1(BG)
        fea_BG = self.calayer_group(fea_BG)
        fea_BG = self.conv_group_2(fea_BG)
        fea_BG = self.conv_group_3(fea_BG)

        fea_FG = self.conv_center_1(FG)
        fea_FG = self.calayer_center(fea_FG)
        fea_FG = self.conv_center_2(fea_FG)

        style = self.conv_shared(torch.cat([fea_BG, fea_FG], dim=1))
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)

        b, c, h, w = fea_BG.size()
        fea_FG = fea_FG.view(b, c, h * w)
        fea_FG_mean = torch.mean(fea_FG, dim=-1, keepdim=True).unsqueeze(3)
        fea_FG_std = torch.std(fea_FG, dim=-1, keepdim=True).unsqueeze(3)

        gamma = gamma + fea_FG_std
        beta = beta + fea_FG_mean

        out = FG_normed * gamma + beta

        ####################################################################
        # 加一个non-local
        mask_center = F.interpolate(mask_center, scale_factor=1/4, mode='bilinear', align_corners=False)
        # print('mask.shape',mask_center.shape)
        detail = self.conv_detail_1(torch.cat([out, fea_BG], dim=1))
        # print(detail.shape)
        detail = self.conv_detail_2(detail)
        detail = self.conv_detail_3(detail)
        detail = detail * mask_center + out
        detail = self.conv_detail_4(detail)
        return out



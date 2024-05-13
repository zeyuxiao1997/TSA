#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# here put the import lib

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
from models.dcn import ModulatedDeformConvPack, modulated_deform_conv

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


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


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

class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(
                f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)

class PAM(nn.Module):  # stereo attention block
    def __init__(self, embed_ch):
        super(PAM, self).__init__()
        self.conv_l = nn.Conv2d(embed_ch, embed_ch, 1, 1, 0, bias=True)
        self.conv_r = nn.Conv2d(embed_ch, embed_ch, 1, 1, 0, bias=True)
        self.rb = ResBlock(embed_ch)
        self.softmax = nn.Softmax(-1)
        self.conv_out = nn.Conv2d(embed_ch * 2 + 1, embed_ch, 1, 1, 0, bias=True)

    def forward(self, x_left, x_right):  # B * C * H * W
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)
        # M_{right_to_left}
        F0 = self.conv_l(buffer_left).permute(0, 2, 3, 1)  # B * H * W * C
        F1 = self.conv_r(buffer_right).permute(0, 2, 1, 3)  # B * H * C * W
        S = torch.bmm(F0.contiguous().view(-1, w, c),
                      F1.contiguous().view(-1, c, w))  # (B*H) * W * W
        # view相当于reshape：若之前进行了permute，必须先进行contiguous
        # bmm相当于矩阵相乘，仅对后两维进行运算，并仅对三维torch进行运算。

        M_right_to_left = self.softmax(S)  # (B*H) * W * W
        # right map transfer to left
        S_T = S.permute(0, 2, 1)  # (B*H) * W * W
        M_left_to_right = self.softmax(S_T)

        # valid mask for transfer information from Feature(right->left)
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1  # (B*H) * 1 * W
        V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)  # 形态学处理


        # V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
        # V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
        # V_right_to_left = morphologic_process(V_right_to_left)

        out_f = x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        out_f = torch.bmm(M_right_to_left, out_f).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)
        # B * C * H * W

        return out_f, V_left_to_right, (M_left_to_right, M_right_to_left)

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'
    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
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

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat=64, kernel_size=3, reduction=8,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size,padding=(kernel_size//2), bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class RCAB_SEACB(nn.Module):
    def __init__(
        self, n_feat=64, kernel_size=3, reduction=8,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB_SEACB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(seacb_conv(n_feat, n_feat))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

## Residual Channel Attention Block (RCAB)
class RCAB_MSCAM(nn.Module):
    def __init__(
            self, n_feat=64, kernel_size=3, reduction=8,
            bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(RCAB_MSCAM, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(MS_CAM(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Conv_19(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_19, self).__init__()
        self.mask = nn.Parameter(torch.Tensor(get_mask_19(in_channels, out_channels)), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 3, 3), requires_grad=True)
        self.weight.data.mul_(self.mask.data)

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)
        x = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1, groups=1)
        return x

def get_mask_19(in_channels, out_channels, kernel_size=3):
    mask = np.zeros((out_channels, in_channels, 3, 3))
    for c in range(kernel_size):
        mask[:, :, c, c] = 1.
    return mask


class Conv_37(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_37, self).__init__()
        self.mask = nn.Parameter(torch.Tensor(get_mask_37(in_channels, out_channels)), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 3, 3), requires_grad=True)
        self.weight.data.mul_(self.mask.data)

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)
        x = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1, groups=1)
        return x

def get_mask_37(in_channels, out_channels, kernel_size=3):
    mask = np.zeros((out_channels, in_channels, 3, 3))
    for c in range(kernel_size):
        mask[:, :, 2-c, c] = 1.
    return mask


class SEACB(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super(SEACB, self).__init__()

        self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=1,
                                     padding=1, bias=False, padding_mode=padding_mode)

        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(3, 1), stride=1,
                                  padding=(1, 0), bias=False, padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(1, 3), stride=1, 
                                  padding=(0, 1), bias=False, padding_mode=padding_mode)

    def forward(self, input):
        square_outputs = self.square_conv(input)
        vertical_outputs = self.ver_conv(input)
        horizontal_outputs = self.hor_conv(input)

        return square_outputs + vertical_outputs + horizontal_outputs


class EACB(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super(EACB, self).__init__()


        self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=1,
                                     padding=1, bias=False, padding_mode=padding_mode)

        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(3, 1), stride=1,
                                  padding=(1, 0), bias=False, padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(1, 3), stride=1, 
                                  padding=(0, 1), bias=False, padding_mode=padding_mode)

        self.diag19_conv = Conv_19(in_channels=in_channels, out_channels=out_channels)

        self.diag37_conv = Conv_37(in_channels=in_channels, out_channels=out_channels)


    def forward(self, input):
        square_outputs = self.square_conv(input)
        vertical_outputs = self.ver_conv(input)
        horizontal_outputs = self.hor_conv(input)
        diag19_outputs = self.diag19_conv(input)
        diag37_outputs = self.diag37_conv(input)

        return square_outputs + vertical_outputs + horizontal_outputs + diag19_outputs + diag37_outputs


def seacb_conv(in_channels, out_channels, kernel_size=3, bias=False):
    return SEACB(in_channels, out_channels, padding_mode='zeros')

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class ResWTBlock(nn.Module):
    def __init__(self, embed_ch):
        super(ResWTBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_ch, embed_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(embed_ch, embed_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.DWT = DWT()
        self.IWT = IWT()
        self.conv = nn.Conv2d(embed_ch, embed_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(embed_ch*2, embed_ch, kernel_size=3, stride=1, padding=1, bias=False)

    def __call__(self, x):
        res = self.body(x)
        resblock = res + x

        _, wt = self.DWT(x)
        wt = self.IWT(wt)
        fea = self.conv(x)
        wtblock = wt+fea

        out = self.conv2(torch.cat((resblock,wtblock),1))
        return out
        
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class CCALayer(nn.Module):#############################################3 new
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(CCALayer, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.conv_du = nn.Sequential(
            nn.Conv2d(nf, 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, nf, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # initialization
        initialize_weights([self.conv1, self.conv_du], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        out = self.contrast(out)+self.avg_pool(out)
        out_channel = self.conv_du(out)
        out_channel = out_channel*out
        out_last = out_channel+identity

        return out_last

class ProgressiveFusion_Block_START_5F(nn.Module): #############################################################################################
    def __init__(self, nf=64):
        super(ProgressiveFusion_Block_START_5F, self).__init__()
        self.conv_encoder = nn.Sequential(nn.Conv2d(3, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True))
        self.fusion = nn.Conv2d(nf*5, nf, 1, 1, 0)
        self.conv_decoder = nn.Sequential(nn.Conv2d(2*nf, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True))
        initialize_weights([self.conv_decoder,self.conv_decoder,self.fusion],0.1)


    def forward(self, x):
        x0 ,x1, x2, x3, x4 = x[:,0,:,:,:],x[:,1,:,:,:],x[:,2,:,:,:],x[:,3,:,:,:],x[:,4,:,:,:]
        x0 = self.conv_encoder(x0)
        x1 = self.conv_encoder(x1)
        x2 = self.conv_encoder(x2)
        x3 = self.conv_encoder(x3)
        x4 = self.conv_encoder(x4)
        x_fusion = self.fusion(torch.cat([x0,x1,x2,x3,x4],1))
        x0 = self.conv_decoder(torch.cat([x0,x_fusion],1))
        x1 = self.conv_decoder(torch.cat([x1, x_fusion], 1))
        x2 = self.conv_decoder(torch.cat([x2, x_fusion], 1))
        x3 = self.conv_decoder(torch.cat([x3, x_fusion], 1))
        x4 = self.conv_decoder(torch.cat([x4, x_fusion], 1))
        x_out = torch.cat([x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),x4.unsqueeze(1)],1)

        return x_out


class ProgressiveFusion_Block_5F(nn.Module): #############################################################################################
    def __init__(self, nf):
        super(ProgressiveFusion_Block_5F, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.fusion = nn.Conv2d(nf*5, nf, 1, 1, 0)
        self.conv_decoder = nn.Sequential(nn.Conv2d(2*nf, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True))
        initialize_weights([self.conv_decoder,self.conv_decoder,self.fusion],0.1)

    def forward(self, x):
        x0_in ,x1_in, x2_in, x3_in, x4_in = x[:,0,:,:,:],x[:,1,:,:,:],x[:,2,:,:,:],x[:,3,:,:,:],x[:,4,:,:,:]
        x0 = self.conv_encoder(x0_in)
        x1 = self.conv_encoder(x1_in)
        x2 = self.conv_encoder(x2_in)
        x3 = self.conv_encoder(x3_in)
        x4 = self.conv_encoder(x4_in)
        x_fusion = self.fusion(torch.cat([x0,x1,x2,x3,x4],1))
        x0 = self.conv_decoder(torch.cat([x0,x_fusion],1))+x0_in
        x1 = self.conv_decoder(torch.cat([x1, x_fusion], 1))+x1_in
        x2 = self.conv_decoder(torch.cat([x2, x_fusion], 1))+x2_in
        x3 = self.conv_decoder(torch.cat([x3, x_fusion], 1))+x3_in
        x4 = self.conv_decoder(torch.cat([x4, x_fusion], 1))+x4_in
        x_out = torch.cat([x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),x4.unsqueeze(1)],1)

        return x_out


class ProgressiveFusion_Block_START_4F(nn.Module): #############################################################################################
    def __init__(self, nf=64):
        super(ProgressiveFusion_Block_START_4F, self).__init__()
        self.conv_encoder = nn.Sequential(nn.Conv2d(3, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True))
        self.fusion = nn.Conv2d(nf*4, nf, 1, 1, 0)
        self.conv_decoder = nn.Sequential(nn.Conv2d(2*nf, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True))
        initialize_weights([self.conv_decoder,self.conv_decoder,self.fusion],0.1)


    def forward(self, x):
        x0 ,x1, x2, x3 = x[:,0,:,:,:],x[:,1,:,:,:],x[:,2,:,:,:],x[:,3,:,:,:]
        x0 = self.conv_encoder(x0)
        x1 = self.conv_encoder(x1)
        x2 = self.conv_encoder(x2)
        x3 = self.conv_encoder(x3)
        x_fusion = self.fusion(torch.cat([x0,x1,x2,x3],1))
        x0 = self.conv_decoder(torch.cat([x0,x_fusion],1))
        x1 = self.conv_decoder(torch.cat([x1, x_fusion], 1))
        x2 = self.conv_decoder(torch.cat([x2, x_fusion], 1))
        x3 = self.conv_decoder(torch.cat([x3, x_fusion], 1))
        x_out = torch.cat([x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1)],1)

        return x_out

class ProgressiveFusion_Block_4F(nn.Module): #############################################################################################
    def __init__(self, nf):
        super(ProgressiveFusion_Block_4F, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.fusion = nn.Conv2d(nf*4, nf, 1, 1, 0)
        self.conv_decoder = nn.Sequential(nn.Conv2d(2*nf, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True))
        initialize_weights([self.conv_decoder,self.conv_decoder,self.fusion],0.1)

    def forward(self, x):
        x0_in ,x1_in, x2_in, x3_in = x[:,0,:,:,:],x[:,1,:,:,:],x[:,2,:,:,:],x[:,3,:,:,:]
        x0 = self.conv_encoder(x0_in)
        x1 = self.conv_encoder(x1_in)
        x2 = self.conv_encoder(x2_in)
        x3 = self.conv_encoder(x3_in)
        x_fusion = self.fusion(torch.cat([x0,x1,x2,x3],1))
        x0 = self.conv_decoder(torch.cat([x0,x_fusion],1))+x0_in
        x1 = self.conv_decoder(torch.cat([x1, x_fusion], 1))+x1_in
        x2 = self.conv_decoder(torch.cat([x2, x_fusion], 1))+x2_in
        x3 = self.conv_decoder(torch.cat([x3, x_fusion], 1))+x3_in
        x_out = torch.cat([x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1)],1)

        return x_out

class ProgressiveFusion_Block_4F_RCAB(nn.Module): #############################################################################################
    def __init__(self, nf):
        super(ProgressiveFusion_Block_4F_RCAB, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.fusion = nn.Conv2d(nf*4, nf, 1, 1, 0)
        self.conv_decoder = nn.Sequential(nn.Conv2d(2*nf, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True),
                                          RCAB(n_feat=nf))
        # initialize_weights([self.conv_decoder,self.conv_decoder,self.fusion],0.1)
        initialize_weights([self.conv_decoder,self.fusion],0.1)

    def forward(self, x):
        x0_in ,x1_in, x2_in, x3_in = x[:,0,:,:,:],x[:,1,:,:,:],x[:,2,:,:,:],x[:,3,:,:,:]
        x0 = self.conv_encoder(x0_in)
        x1 = self.conv_encoder(x1_in)
        x2 = self.conv_encoder(x2_in)
        x3 = self.conv_encoder(x3_in)
        x_fusion = self.fusion(torch.cat([x0,x1,x2,x3],1))
        x0 = self.conv_decoder(torch.cat([x0,x_fusion],1))+x0_in
        x1 = self.conv_decoder(torch.cat([x1, x_fusion], 1))+x1_in
        x2 = self.conv_decoder(torch.cat([x2, x_fusion], 1))+x2_in
        x3 = self.conv_decoder(torch.cat([x3, x_fusion], 1))+x3_in
        x_out = torch.cat([x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1)],1)

        return x_out


class ProgressiveFusion_Block_7F(nn.Module): #############################################################################################
    def __init__(self, nf):
        super(ProgressiveFusion_Block_7F, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.fusion = nn.Conv2d(nf*7, nf, 1, 1, 0)
        self.conv_decoder = nn.Sequential(nn.Conv2d(2*nf, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True))
        initialize_weights([self.conv_decoder,self.conv_decoder,self.fusion],0.1)

    def forward(self, x):
        x0_in ,x1_in, x2_in, x3_in,x4_in, x5_in, x6_in = x[:,0,:,:,:],x[:,1,:,:,:],x[:,2,:,:,:],x[:,3,:,:,:],x[:,4,:,:,:],x[:,5,:,:,:],x[:,6,:,:,:]
        x0 = self.conv_encoder(x0_in)
        x1 = self.conv_encoder(x1_in)
        x2 = self.conv_encoder(x2_in)
        x3 = self.conv_encoder(x3_in)
        x4 = self.conv_encoder(x4_in)
        x5 = self.conv_encoder(x5_in)
        x6 = self.conv_encoder(x6_in)
        x_fusion = self.fusion(torch.cat([x0,x1,x2,x3,x4,x5,x6],1))
        x0 = self.conv_decoder(torch.cat([x0, x_fusion], 1))+x0_in
        x1 = self.conv_decoder(torch.cat([x1, x_fusion], 1))+x1_in
        x2 = self.conv_decoder(torch.cat([x2, x_fusion], 1))+x2_in
        x3 = self.conv_decoder(torch.cat([x3, x_fusion], 1))+x3_in
        x4 = self.conv_decoder(torch.cat([x4, x_fusion], 1))+x4_in
        x5 = self.conv_decoder(torch.cat([x5, x_fusion], 1))+x5_in
        x6 = self.conv_decoder(torch.cat([x6, x_fusion], 1))+x6_in
        x_out = torch.cat([x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),x4.unsqueeze(1),x5.unsqueeze(1),x6.unsqueeze(1)],1)

        return x_out

class ProgressiveFusion_Block_7F_RCAB(nn.Module): #############################################################################################
    def __init__(self, nf):
        super(ProgressiveFusion_Block_7F_RCAB, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.fusion = nn.Conv2d(nf*7, nf, 1, 1, 0)
        self.conv_decoder = nn.Sequential(nn.Conv2d(2*nf, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True),
                                          RCAB(n_feat=nf))
        # initialize_weights([self.conv_decoder,self.conv_decoder,self.fusion],0.1)
        initialize_weights([self.conv_decoder,self.fusion],0.1)

    def forward(self, x):
        x0_in ,x1_in, x2_in, x3_in,x4_in, x5_in, x6_in = x[:,0,:,:,:],x[:,1,:,:,:],x[:,2,:,:,:],x[:,3,:,:,:],x[:,4,:,:,:],x[:,5,:,:,:],x[:,6,:,:,:]
        x0 = self.conv_encoder(x0_in)
        x1 = self.conv_encoder(x1_in)
        x2 = self.conv_encoder(x2_in)
        x3 = self.conv_encoder(x3_in)
        x4 = self.conv_encoder(x4_in)
        x5 = self.conv_encoder(x5_in)
        x6 = self.conv_encoder(x6_in)
        x_fusion = self.fusion(torch.cat([x0,x1,x2,x3,x4,x5,x6],1))
        x0 = self.conv_decoder(torch.cat([x0, x_fusion], 1))+x0_in
        x1 = self.conv_decoder(torch.cat([x1, x_fusion], 1))+x1_in
        x2 = self.conv_decoder(torch.cat([x2, x_fusion], 1))+x2_in
        x3 = self.conv_decoder(torch.cat([x3, x_fusion], 1))+x3_in
        x4 = self.conv_decoder(torch.cat([x4, x_fusion], 1))+x4_in
        x5 = self.conv_decoder(torch.cat([x5, x_fusion], 1))+x5_in
        x6 = self.conv_decoder(torch.cat([x6, x_fusion], 1))+x6_in
        x_out = torch.cat([x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),x4.unsqueeze(1),x5.unsqueeze(1),x6.unsqueeze(1)],1)

        return x_out



class Conv_relu(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size, stride, padding, has_relu=True, efficient=False):
        super(Conv_relu, self).__init__()
        self.has_relu = has_relu
        self.efficient = efficient

        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        def _func_factory(conv, relu, has_relu):
            def func(x):
                x = conv(x)
                if has_relu:
                    x = relu(x)
                return x
            return func

        func = _func_factory(self.conv, self.relu, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x

class Attention(nn.Module):
    def __init__(self, nf=64):
        super(Attention, self).__init__()
        self.sAtt_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        att = self.lrelu(self.sAtt_1(x))
        att_max = self.max_pool(att)
        att_avg = self.avg_pool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))

        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.max_pool(att_L)
        att_avg = self.avg_pool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        out = x * att * 2 + att_add

        return out

class CNCAM(nn.Module):
    def __init__(self, nf=64, n_level=3):
        super(CNCAM, self).__init__()
        self.nf = nf
        self.nl = n_level
        self.down_conv = Conv_relu(nf, nf, 3, 2, 1, has_relu=True)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.cat_conv = Conv_relu(nf * self.nl + nf, nf * 4, 1, 1, 0, has_relu=True)
        self.ps = nn.PixelShuffle(2)
        self.up_conv1 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.up_conv2 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)

        self.spa_l = nn.ModuleList()
        for i in range(self.nl + 1):
            self.spa_l.append(Attention(nf))

    def forward(self, x):
        down_fea = self.down_conv(x)
        B, C, H, W = down_fea.size()
        ind_B = torch.arange(B, dtype=torch.long).repeat_interleave(H * W).cuda()

        p_list = list()
        for j in range(self.nl):
            if p_list:
                p_list.append(self.avg_pool(p_list[-1]))
            else:
                p_list.append(self.avg_pool(down_fea))

        # size of query: B, H * W, C
        query = down_fea.view(B, C, H * W).permute(0, 2, 1)
        query = F.normalize(query, p=2, dim=2)

        keys = list()
        for j in range(self.nl):
            keys.append(F.normalize(p_list[j].view(B, C, -1), p=2, dim=1))

        att_fea = self.spa_l[0](down_fea)
        all_f = [att_fea]
        for j in range(self.nl):
            sim = torch.matmul(query, keys[j])
            ind = sim.argmax(dim=2).view(-1)
            sim_f = keys[j][ind_B, :, ind].view(B, H, W, C).permute(0, 3, 1, 2)
            att_sim_f = self.spa_l[j + 1](sim_f)
            all_f.append(att_sim_f)

        all_f = torch.cat(all_f, dim=1)
        cat_fea = self.cat_conv(all_f)
        up_fea = self.ps(cat_fea)
        up_fea = self.up_conv1(up_fea)

        fea = torch.cat([x, up_fea], dim=1)
        out = self.up_conv2(fea)

        return out


class TemporalFusion(nn.Module):
    def __init__(self, nf, n_frame):
        super(TemporalFusion, self).__init__()
        self.n_frame = n_frame

        self.ref_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.nbr_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.up_conv = nn.Conv2d(nf * n_frame, nf * 4, 1, 1, bias=True)
        self.ps = nn.PixelShuffle(2)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()

        emb_ref = self.ref_conv(x[:, N // 2, :, :, :].clone())
        emb = self.nbr_conv(x.view(-1, C, H, W)).view(B, N, C, H, W)

        cor_l = []
        for i in range(N):
            cor = torch.sum(emb[:, i, :, :, :] * emb_ref, dim=1, keepdim=True)
            cor_l.append(cor)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aggr_fea = x.view(B, -1, H, W) * cor_prob

        fea = self.lrelu(self.up_conv(aggr_fea))
        out = self.ps(fea)

        return out


class TMCAM(nn.Module):
    def __init__(self, nf, n_frame, nbr, n_group, kernels, patches, cor_ksize):
        super(TMCAM, self).__init__()
        self.n_frame = n_frame

        self.aggr = Aggregate(nf=nf, nbr=nbr, n_group=n_group, kernels=kernels, patches=patches, cor_ksize=cor_ksize)
        self.tf = TemporalFusion(nf=nf, n_frame=n_frame)

    def forward(self, x):
        L1_fea, L2_fea, L3_fea = x
        center = self.n_frame // 2

        ref_fea_l = [
            L1_fea[:, center, :, :, :].clone(), L2_fea[:, center, :, :, :].clone(),
            L3_fea[:, center, :, :, :].clone()
        ]

        aggr_fea_l = []
        for i in range(self.n_frame):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aggr_fea_l.append(self.aggr(nbr_fea_l, ref_fea_l))

        aggr_fea = torch.stack(aggr_fea_l, dim=1)  # [B, N, C, H, W]
        out = self.tf(aggr_fea)

        return out

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self, level, rfb=False):
        super(ASFF, self).__init__()
        self.level = level
        # 输入的三个特征层的channels, 根据实际修改
        self.dim = [64,64,64]
        self.inter_dim = self.dim[self.level]
        # 每个层级三者输出通道数需要一致
        if level==0:
            self.stride_level_1 = add_conv(self.dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 64, 3, 1)
        elif level==1:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 64, 3, 1)
        elif level==2:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            if self.dim[1] != self.dim[2]:
                self.compress_level_1 = add_conv(self.dim[1], self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 64, 3, 1)
        compress_c = 8 if rfb else 16  
        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, 1, 1, 0)

  # 尺度大小 level_0 < level_1 < level_2
    def forward(self, x_level_0, x_level_1, x_level_2):
        # Feature Resizing过程
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')  
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            if self.dim[1] != self.dim[2]:
                level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            else:
                level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2
    # 融合权重也是来自于网络学习
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v,
                                     level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)   # alpha产生
    # 自适应融合
        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)
        return out
        # return level_0_resized



if __name__ == "__main__":
    x1 = torch.rand(1, 64, 64,64).cuda()
    print((x1/2).shape)
    model = ResWTBlock(embed_ch=64).cuda()
    a= model(x1)
    print(a.shape)
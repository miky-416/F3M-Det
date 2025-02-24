import math

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from matplotlib import pyplot as plt
from timm.models.layers import trunc_normal_
from torch.nn import Parameter
from models.common import Conv, MAM1, MAM2, CEM1
# from models.dynamic_conv import DynamicConv, DynamicConv1
from models.spatial_transformer import FFM, CEM
from .FreqFusion import *
from .D2A2 import *


class Fuse1(nn.Module):
    def __init__(self, in_channel):
        super(Fuse1, self).__init__()

        self.mam = MAM2(in_channel)
        self.dda = DDA(in_channel)
        self.ff = FreqFusion1x(in_channel, in_channel)

    def forward(self,x):
        rgb = x[0]
        t = x[1]

        x = [rgb, t]
        gr = self.mam(x)
        dda_gr = self.dda(gr, t)
        final = self.ff(dda_gr, t)

        return final


class Fuse12(nn.Module):
    def __init__(self, in_channel):
        super(Fuse12, self).__init__()

        self.mam = MAM2(in_channel)
        self.dda = DDA(in_channel)
        self.conv_1x1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1)
        self.ff1 = FreqFusion1x(in_channel, in_channel)
        self.ff2 = FreqFusion2x(in_channel, in_channel)

    def forward(self,x):
        rgb = x[0]
        t = x[1]
        last_fea = x[2]
        last_fea = self.conv_1x1(last_fea)

        x = [rgb, t]
        gr = self.mam(x)
        dda_gr = self.dda(gr, t)
        ff_fea = self.ff1(dda_gr, t)
        final = self.ff2(ff_fea, last_fea)
        return final


class DDA(nn.Module):
    def __init__(self, n_feats):
        super(DDA, self).__init__()
        self.conv_fuse1 = conv3x3(n_feats * 2, n_feats)
        self.get_offset = conv3x3(n_feats, n_feats)
        self.DCN = DeformConv(n_feats, n_feats, kernel_size=3, stride=1, padding=1,
                              dilation=1,
                              groups=1,
                              deformable_groups=8, im2col_step=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, rgb, depth):
        res_fea = rgb - depth
        rgb32 = rgb.to(dtype=torch.float)
        offset = self.act(self.conv_fuse1(torch.cat((res_fea, rgb), dim=1)))
        offset = self.act(self.get_offset(offset))
        offset32 = offset.to(dtype=torch.float)
        rgb_guided = self.DCN(rgb32, offset32)

        return rgb_guided + rgb32


class CEMFRM(nn.Module):
    def __init__(self, in_channel):
        super(CEMFRM, self).__init__()

        self.cem = CEM1(in_channel)
        self.frm = FRM(in_channel)

    def forward(self, x):
        rgb = x[0]
        t = x[1]
        # t1 = self.esam(t)
        # rgb1 = self.DSMM(rgb,t)
        x = [rgb, t]
        cem_rgb, cem_ir = self.cem(rgb, t)
        frm_rgb, frm_ir = self.frm(rgb, t)
        rgb_out = cem_rgb + frm_rgb
        ir_out = cem_ir + frm_ir

        return rgb_out, ir_out


class FRM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FRM, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
    # def forward(self, x):
    #     x1 = x[0]
    #     x2 = x[1]
        channel_weights = self.channel_weights(x1, x2)
        # out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        # x1 = x1 + self.lambda_c * channel_weights[0] * x1
        x1 = x1 + channel_weights[0] * x1
        # out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        # x2 = x2 + self.lambda_c * channel_weights[1] * x2
        x2 = x2 + channel_weights[1] * x2

        spatial_weights = self.spatial_weights(x1, x2)
        # out_x1 = x1 + self.lambda_s * spatial_weights[0] * x1
        out_x1 = x1 + spatial_weights[0] * x1
        # out_x2 = x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + spatial_weights[1] * x2

        # out = out_x1 + out_x2
        # return out
        return out_x1, out_x2



# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 2 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        # x = torch.cat((x1, x2), dim=1)
        # avg1 = self.avg_pool(x1).view(B, self.dim)
        avg1 = torch.mean(x1, dim=[2, 3], keepdim=True).view(B, self.dim)
        avg2 = torch.mean(x2, dim=[2, 3], keepdim=True).view(B, self.dim)
        # avg2 = self.avg_pool(x2).view(B, self.dim)
        max1 = self.max_pool(x1).view(B, self.dim)
        max2 = self.max_pool(x2).view(B, self.dim)
        avg = avg1+avg2
        max = max1+max2
        y = torch.cat((max, avg), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



# Edge-based Enhancement Unit (EEU)
class EEU(nn.Module):
    def __init__(self, in_channel):
        super(EEU, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.PReLU = nn.PReLU(in_channel)

    def forward(self, x):
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        # edge = self.PReLU(edge)
        out = weight * x + x
        return out


# Edge Self-Alignment Module (ESAM)
class ESAM(nn.Module):
    def __init__(self, in_channel):
        super(ESAM, self).__init__()
        self.eeu = EEU(in_channel)
    def forward(self, t):  # x1 16*144*14; x2 24*72*72
        t_2 = self.eeu(t)
        return t_2  # (24*2)*144*144



if __name__ == "__main__":
    x = torch.rand(4,64,80,80).half().cuda()
    y = torch.rand(4,64,80,80).half().cuda()
    m = [x,y]
    fuse = FFuse11(64).half().cuda()
    out = fuse(m)
    print(out.shape)
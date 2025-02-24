import math
import warnings
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.cuda import amp
import torch.nn.functional as F

# from models.CBAM import CBAMBlock
# from models.CoordAttention import CoordAtt
# from models.DCnet import convblock
# from models.DDPM import DDPM
# from models.based_transformer import FRM
# from models.spatial_transformer import CEM, CEM1
from models.spatial_transformer import CEM1
from models.attempt.DCnet import convblock
from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
from torch.nn import init, Sequential
from .CLFT import *
from .LDConv import *
from .assemFormer import AssemFormer
from .BiFormer import BiLevelRoutingAttention_nchw
# from .dynamic import FRM

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class MAM2(nn.Module):
    def __init__(self, in_channel):
        super(MAM2, self).__init__()
        self.channel264 = nn.Sequential(
            Conv(in_channel, in_channel//2, 3, 2, 1),
            convblock(in_channel//2, in_channel//4, 3, 1, 1),
            convblock(in_channel//4, in_channel//8, 3, 1, 0),
            convblock(in_channel//8, in_channel//16, 3, 1, 1),
            convblock(in_channel//16, 16, 1, 1, 1),
        )
        self.xy = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 2, 1, 1, 0)
        )
        self.scale1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, 1, 1, 0)
        )
        self.scale2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, 1, 1, 0)
        )

        self.scale3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, 1, 1, 0)
        )


        # Start with identity transformation
        self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.xy[-1].bias.data.zero_()
        self.scale1[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale1[-1].bias.data.zero_()
        self.scale2[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale2[-1].bias.data.zero_()
        self.scale3[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.scale3[-1].bias.data.zero_()

    def forward(self, x):
        gr = x[0]
        gt = x[1]
        in_ = gt - gr
        n1 = self.channel264(in_)
        # in_ = torch.cat([gr, gt], dim=1)

        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)

        shift_xy = self.xy(n1)
        shift_s1 = self.scale1(n1)
        shift_s2 = self.scale2(n1)
        rotation = self.scale3(n1)

        bsize = shift_xy.shape[0]
        identity_theta = identity_theta.view(-1, 2, 3).repeat(bsize, 1, 1).cuda()

        identity_theta[:, :, 2] = shift_xy.squeeze()
        identity_theta[:, :1, :1] = shift_s1.squeeze(2)
        identity_theta[:, 1, 1] = shift_s2.squeeze()

        # 应用旋转
        cos_theta = rotation.cos()  # 获取 cos(θ)
        sin_theta = rotation.sin()  # 获取 sin(θ)

        # 更新仿射矩阵中的旋转部分
        identity_theta[:, 0, 0] *= cos_theta.squeeze()  # x轴缩放和旋转结合
        identity_theta[:, 0, 1] = -sin_theta.squeeze()  # x 方向的旋转
        identity_theta[:, 1, 0] = sin_theta.squeeze()  # y 方向的旋转
        identity_theta[:, 1, 1] *= cos_theta.squeeze()  # y轴缩放和旋转结合

        wrap_grid = F.affine_grid(identity_theta.view(-1, 2, 3), in_.size(), align_corners=True).permute(0, 3, 1, 2)
        wrap_gr = F.grid_sample(gr, wrap_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros',
                                align_corners=True)

        return wrap_gr


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=4, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_3 = nn.AdaptiveAvgPool2d((3, 3))
        self.avgpool_6 = nn.AdaptiveAvgPool2d((6, 6))
        self.avgpool_8 = nn.AdaptiveAvgPool2d((8, 8))

        self.fc_rgb = nn.Linear(110, vert_anchors*horz_anchors)
        self.fc_ir = nn.Linear(110, vert_anchors*horz_anchors)

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)
        """
        rgb = x[0]
        ir = x[1]
        assert rgb.shape[0] == ir.shape[0]
        bs, c, h, w = rgb.shape
        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea_1 = self.avgpool_1(rgb)
        ir_fea_1 = self.avgpool_1(ir)
        rgb_fea_3 = self.avgpool_3(rgb)
        ir_fea_3 = self.avgpool_3(ir)
        rgb_fea_6 = self.avgpool_6(rgb)
        ir_fea_6 = self.avgpool_6(ir)
        rgb_fea_8 = self.avgpool_8(rgb)
        ir_fea_8 = self.avgpool_8(ir)

        rgb_fea_1_flat = rgb_fea_1.view(bs, c, -1)
        ir_fea_1_flat = ir_fea_1.view(bs, c, -1)
        rgb_fea_3_flat = rgb_fea_3.view(bs, c, -1)
        ir_fea_3_flat = ir_fea_3.view(bs, c, -1)
        rgb_fea_6_flat = rgb_fea_6.view(bs, c, -1)
        ir_fea_6_flat = ir_fea_6.view(bs, c, -1)
        rgb_fea_8_flat = rgb_fea_8.view(bs, c, -1)
        ir_fea_8_flat = ir_fea_8.view(bs, c, -1)

        rgb_fea_flat = torch.cat([rgb_fea_1_flat, rgb_fea_3_flat, rgb_fea_6_flat, rgb_fea_8_flat], dim=2)
        ir_fea_flat = torch.cat([ir_fea_1_flat, ir_fea_3_flat, ir_fea_6_flat, ir_fea_8_flat], dim=2)


        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        rgb_fea_flat = self.fc_rgb(rgb_fea_flat)
        ir_fea_flat = self.fc_ir(ir_fea_flat)

        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat and fc
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out


# class GPT(nn.Module):
#     """  the full GPT language model, with a context size of block_size """
#
#     def __init__(self, d_model, h=8, block_exp=4,
#                  n_layer=4, vert_anchors=13, horz_anchors=13,
#                  embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
#         super().__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))
#
#         # transformer
#         self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
#                                             for layer in range(n_layer)])
#
#         # decoder head
#         self.ln_f = nn.LayerNorm(self.n_embd)
#
#         # regularization
#         self.drop = nn.Dropout(embd_pdrop)
#
#         # avgpool
#         self.avgpool_5 = nn.AdaptiveAvgPool2d((5, 5))
#         self.avgpool_12 = nn.AdaptiveAvgPool2d((12, 12))
#
#         # init weights
#         self.apply(self._init_weights)
#
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, x):
#         """
#         Args:
#             x (tuple?)
#         """
#         rgb = x[0]
#         ir = x[1]
#         assert rgb.shape[0] == ir.shape[0]
#         bs, c, h, w = rgb.shape
#         # -------------------------------------------------------------------------
#         # AvgPooling
#         # -------------------------------------------------------------------------
#         # AvgPooling for reduce the dimension due to expensive computation
#         rgb_fea_5 = self.avgpool_5(rgb)
#         ir_fea_5 = self.avgpool_5(ir)
#         rgb_fea_12 = self.avgpool_12(rgb)
#         ir_fea_12 = self.avgpool_12(ir)
#
#         rgb_fea_5_flat = rgb_fea_5.view(bs, c, -1)
#         ir_fea_5_flat = ir_fea_5.view(bs, c, -1)
#         rgb_fea_12_flat = rgb_fea_12.view(bs, c, -1)
#         ir_fea_12_flat = ir_fea_12.view(bs, c, -1)
#
#         rgb_fea_flat = torch.cat([rgb_fea_5_flat, rgb_fea_12_flat], dim=2)
#         ir_fea_flat = torch.cat([ir_fea_5_flat, ir_fea_12_flat], dim=2)
#
#         # -------------------------------------------------------------------------
#         # Transformer
#         # -------------------------------------------------------------------------
#
#         token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat and fc
#         token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)
#
#         # transformer
#         x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
#         x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)
#
#         # decoder head
#         x = self.ln_f(x)  # dim:(B, 2*H*W, C)
#         x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
#         x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
#
#         # 这样截取的方式, 是否采用映射的方式更加合理？
#         rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
#         ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
#
#         # -------------------------------------------------------------------------
#         # Interpolate (or Upsample)
#         # -------------------------------------------------------------------------
#         rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear', align_corners=False)
#         ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear', align_corners=False)
#
#         return rgb_fea_out, ir_fea_out



# Github地址：https://github.com/JIAOJIAYUASD/dilateformer
# 论文地址：https://arxiv.org/abs/2302.01791
class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[2, 3]):
        super().__init__()
        self.dim = dim *2
        self.num_heads = num_heads
        head_dim = (dim*2) // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim*2, dim *2*3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim*2, dim*2)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        rgb = x[0]
        ir = x[1]
        y = torch.cat((rgb, ir), dim = 1)
        B, H, W, C = y.shape
        # x = y.permute(0, 3, 1, 2)  # B, C, H, W
        qkv = self.qkv(y).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # num_dilation,3,B,C//num_dilation,H,W
        y = y.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            y[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
        x = y.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x + rgb, x + ir


# Github地址：https://github.com/hhb072/SViT
# 论文地址：https://arxiv.org/pdf/2211.11167
class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size = [8, 8], n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.stoken_size = stoken_size

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=proj_drop)

    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape

        hh, ww = H // h, W // w

        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))  # (B, C, hh, ww)

        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale  # (B, hh*ww, h*w, 9)

                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)

                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)

                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)

                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)

        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)

        stoken_features = self.stoken_refine(stoken_features)

        stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)

        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)

        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features


    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)



############ PPA + CEM1 module #####################
class PPA(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()

        self.skip = conv_block(in_features=in_features,
                               out_features=in_features,
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               norm_type='bn',
                               activation=False)
        self.c1 = conv_block(in_features=in_features,
                             out_features=in_features,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c2 = conv_block(in_features=in_features,
                             out_features=in_features,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.c3 = conv_block(in_features=in_features,
                             out_features=in_features,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             norm_type='bn',
                             activation=True)
        self.sa = SpatialAttentionModule()
        self.cn = ECA(in_features)
        self.lga2 = LocalGlobalAttention(in_features, 2)
        self.lga4 = LocalGlobalAttention(in_features, 4)

        self.bn1 = nn.BatchNorm2d(in_features)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

        #与原作者一样，再加一个相同的CEM1
        self.cem = CEM1(in_features)


    def forward(self, x):
        rgb = x[0]
        ir = x[1]
        input = ir - rgb
        #PPA
        x_skip = self.skip(input)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(input)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)

        cem_rgb, cem_ir = self.cem(rgb, ir)
        out_rgb = x + cem_rgb
        out_ir = x + cem_ir
        return out_rgb, out_ir


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x
#############################################




class NAFSCAM(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        self.blk_rgb = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.blk_ir = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c)

    def forward(self, x):
        rgb = x[0]
        ir = x[1]
        rgb_fea = self.blk_rgb(rgb)
        ir_fea = self.blk_ir(ir)
        feats = [rgb_fea, ir_fea]
        feats = self.fusion(feats)

        return feats


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma



class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    # def forward(self, x_l, x_r):
    def forward(self, x):
        x_l = x[0]
        x_r = x[1]
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r



class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2




class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            # final = torch.add(x[0], x[1][0])
            # final1 = torch.add(x[0], x[1][1])
            # final2 = torch.add(final, final1)
            # map_rgb = torch.unsqueeze(torch.mean(final2, 1), 1)
            # score2 = F.interpolate(map_rgb, size=(40, 40), mode="bilinear", align_corners=True)
            # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
            # depth = (score2 - score2.min()) / (score2.max() - score2.min())
            # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
            # plt.imshow(feature_img)
            # plt.show()
            # plt.savefig("29.png")
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            out = torch.add(x[0], x[1][1])
            # print(x[0].size(),x[1][1].size())
            # map_rgb = torch.unsqueeze(torch.mean(out, 1), 1)
            # score2 = F.interpolate(map_rgb, size=(40, 40), mode="bilinear", align_corners=True)
            # score2 = np.squeeze(torch.sigmoid(score2).cpu().data.numpy())
            # depth = (score2 - score2.min()) / (score2.max() - score2.min())
            # feature_img = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_JET)
            # plt.imshow(feature_img)
            # plt.show()
            # plt.savefig("30.png")
            return out

        # return torch.add(x[0], x[1])


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        res= self.act(self.bn(self.conv(x)))
        return res

    def fuseforward(self, x):
        res = self.act(self.conv(x))

        return res


class Conv1(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv1, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        if isinstance(x, list):
            # 如果 x 是列表，使用第一个元素
            y = x[0]
            res = self.act(self.bn(self.conv(y)))
        else:
            # 如果 x 不是列表，直接使用 x
            res = self.act(self.bn(self.conv(x)))
        return res

    def fuseforward(self, x):
        if isinstance(x, list):
            # 如果 x 是列表，使用第一个元素
            res = self.act(self.conv(x[0]))
        else:
            # 如果 x 不是列表，直接使用 x
            res = self.act(self.conv(x))

        return res


class Concat1(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat1, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        if isinstance(x[1], list):
            x0 = x[0]
            x1 = x[1][0]
            y = [x0, x1]
            return torch.cat(y, self.d)
        else:
            return torch.cat(x, self.d)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        final = self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        return final
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        final = torch.add(x[0], x[1])
        return final



class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)



class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop_t = nn.Dropout(attn_pdrop)
        self.attn_drop_s = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        q_s, q_t = torch.chunk(q,2,2)
        k_s, k_t = torch.chunk(k,2,3)
        v_s, v_t = torch.chunk(v,2,2)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        # att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att_s = torch.matmul(q_s, k_t) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        # att_s = torch.matmul(q_s, k_s) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        att_t = torch.matmul(q_t, k_t) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        # att_t = torch.matmul(q_t, k_s) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        # if attention_weights is not None:
        #     att = att * attention_weights
        # if attention_mask is not None:
        #     att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att_s = torch.softmax(att_s, -1)
        att_s = self.attn_drop_s(att_s)

        att_t = torch.softmax(att_t, -1)
        att_t = self.attn_drop_t(att_t)

        # output
        # out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_t = torch.matmul(att_t, v_t).permute(0, 2, 1, 3).contiguous().view(b_s, nq//2, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_s = torch.matmul(att_s, v_s).permute(0, 2, 1, 3).contiguous().view(b_s, nq//2, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out1 = torch.cat([out_s, out_t], dim=1)
        out = self.resid_drop(self.out_proj(out1))  # (b_s, nq, d_model)
        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model,d_k,d_v,h)
        # self.sa = Attention(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()
        # x = x + self.sa(self.ln_input(x))
        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))
        return x


# class SACE(nn.Module):
#     def __init__(self, d_model, h=8, block_exp=4,
#                  n_layer=2, vert_anchors=8, horz_anchors=8,
#                  embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
#         super().__init__()
#
#         self.n_embd = d_model
#         self.vert_anchors = vert_anchors
#         self.horz_anchors = horz_anchors
#
#         d_k = d_model
#         d_v = d_model
#
#         # positional embedding parameter (learnable), rgb_fea + ir_fea
#         self.pos_emb_rgb = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
#
#         # transformer
#         self.trans_blocks_r = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
#                                             for layer in range(n_layer)])
#         # decoder head
#         self.ln_f_r = nn.LayerNorm(self.n_embd)
#
#         # regularization
#         self.drop_r = nn.Dropout(embd_pdrop)
#
#         # avgpool
#         self.avgpool_r = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
#
#         # init weights
#         self.apply(self._init_weights)
#         self.cem1 = CEM(d_model)
#     @staticmethod
#     def _init_weights(module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self, rgb):
#         bs, c, h, w = rgb.shape
#         # -------------------------------------------------------------------------
#         # AvgPooling
#         # -------------------------------------------------------------------------
#         # AvgPooling for reduce the dimension due to expensive computation
#         rgb_fea = self.avgpool_r(rgb)
#         # -------------------------------------------------------------------------
#         # Transformer
#         # -------------------------------------------------------------------------
#         # pad token embeddings along number of tokens dimension
#         rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
#
#
#         token_embeddings_rgb = rgb_fea_flat.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)
#
#         # transformer
#         x_rgb = self.drop_r(self.pos_emb_rgb + token_embeddings_rgb)  # sum positional embedding and token    dim:(B, 2*H*W, C)
#         x_rgb = self.trans_blocks_r(x_rgb)  # dim:(B, 2*H*W, C)
#         x_rgb = self.ln_f_r(x_rgb)  # dim:(B, 2*H*W, C)
#         x_rgb = x_rgb.view(bs,self.vert_anchors, self.horz_anchors, self.n_embd)
#         x_rgb = x_rgb.permute(0, 3, 1, 2)  # dim:(B, C, H, W)
#
#         # -------------------------------------------------------------------------
#         # Interpolate (or Upsample)
#         # -------------------------------------------------------------------------
#         rgb_fea_out = F.interpolate(x_rgb, size=([h, w]), mode='bilinear')
#         rgb_c = self.cem1(rgb)
#         out1 = torch.add(rgb_fea_out, rgb_c)
#         return out1


# class SACE1(nn.Module):
#     def __init__(self,in_channel):
#         super(SACE1, self).__init__()
#         self.sace1 = SACE(in_channel)
#         self.sace2 = SACE(in_channel)
#     def forward(self,x):
#         rgb = x[0]
#         t = x[1]
#         rgb_out = self.sace1(rgb)
#         t_out = self.sace1(t)
#         return rgb_out, t_out


def kernel2d_conv(feat_in, kernel, ksize):
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        # 版本差异导致init报错的，可以在init前面加上nn.
        nn.init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        nn.init.normal_(self.rel_h, 0, 1)
        nn.init.normal_(self.rel_w, 0, 1)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class MAM(nn.Module):
    def __init__(self, in_channel):
        super(MAM, self).__init__()
        self.channel264 = nn.Sequential(
            Conv(in_channel*2, in_channel, 3, 1, 1),
            Conv(in_channel, in_channel//2, 3, 1, 1),
            Conv(in_channel//2, in_channel//4, 3, 1, 1),
            Conv(in_channel//4, 64, 1, 1, 0)
        )
        self.xy = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 2, 1, 1, 0)
        )
        # Start with identity transformation
        self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
        self.xy[-1].bias.data.zero_()
        self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
    def forward(self, x):
        gr = x[0]
        gt = x[1]
        in_ = torch.cat([gr, gt], dim=1)
        n1 = self.channel264(in_)
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
        if in_.is_cuda:
            identity_theta = identity_theta.cuda().detach()
        shift_xy = self.xy(n1)
        bsize = shift_xy.shape[0]
        identity_theta = identity_theta.view(-1, 2, 3).repeat(bsize, 1, 1)
        identity_theta[:, :, 2] += shift_xy.squeeze()
        identity_theta = identity_theta.half()
        wrap_grid = F.affine_grid(identity_theta.view(-1, 2, 3), in_.size(), align_corners=True).permute(0, 3, 1, 2)
        wrap_gr = F.grid_sample(gr, wrap_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros',align_corners=True)
        feature_fuse1 = self.fus1(torch.cat([wrap_gr, gt], dim=1))
        return feature_fuse1


# class MAM(nn.Module):
#     def __init__(self, in_channel):
#         super(MAM, self).__init__()
#         self.channel264 = nn.Sequential(
#             Conv(in_channel*2, in_channel, 3, 1, 1),
#             Conv(in_channel, in_channel//2, 3, 1, 1),
#             Conv(in_channel//2, in_channel//4, 3, 1, 1),
#             Conv(in_channel//4, 64, 1, 1, 0)
#         )
#         self.deta = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(64, 6, 1, 1, 0)
#         )
#         # Start with identity transformation
#         self.deta[-1].weight.data.normal_(mean=0.0, std=5e-4)
#         self.deta[-1].bias.data.zero_()
#         self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
#         # self.refine = Fuser(in_channel)
#         # self.fuse = nn.Conv2d(in_channel,in_channel,1,1)
#         # self.gpt = GPT(in_channel)
#         self.ddpm = DDPM(in_channel,in_channel,in_channel)
#     def forward(self, x):
#         gr = x[0]
#         gt = x[1]
#         # in_ = torch.cat([gr, gt], dim=1)
#         # n1 = self.channel264(in_)
#         # identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False).half()
#         # if in_.is_cuda:
#         #     identity_theta = identity_theta.cuda().detach()
#         # deta = self.deta(n1)
#         # bsize = deta.shape[0]
#         # a = identity_theta.unsqueeze(0).repeat(bsize, 1)
#         # affine_matrix = (deta.view(bsize, -1) + a)
#         # wrap_grid = F.affine_grid(affine_matrix.view(-1, 2, 3), in_.size(), align_corners=True).permute(0, 3, 1,2)
#         # wrap_gr = F.grid_sample(gr, wrap_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros',align_corners=True)
#         # feature_fuse1 = self.fus1(torch.cat([wrap_gr, gt], dim=1))
#         fuse = self.ddpm(gr, gt)
#         return fuse


class CM(nn.Module):
    def __init__(self,in_channel):
        super(CM, self).__init__()
        # self.alpha = nn.Conv2d(in_channel, 1, 1, 1, 0)
        # self.bata = nn.Conv2d(in_channel, 1, 1, 1, 0)
        # self.fus = Conv(in_channel, in_channel, 1, 1, 0)
        self.dynamic_filter = nn.Conv2d(in_channel, 3 * 3 * in_channel, 3, 1, 1)
        # self.fus = Conv(in_channel, in_channel, 1, 1, 0)
    def forward(self, x):
        gr = x[0]
        gt = x[1]
        # affine_gr = self.alpha(gt)*gr + self.bata(gt)
        # in2 = self.fus(gt+affine_gr)
        filter = self.dynamic_filter(gt)
        in3 = kernel2d_conv(gr, filter, 3)
        return in3


# class DDPM1(nn.Module):
#     def __init__(self, in_channel):
#         super(DDPM1, self).__init__()
#         self.ddpm = DDPM(in_channel,in_channel,in_channel)
#     def forward(self, x):
#         gr = x[0]
#         gt = x[1]
#         fuse = self.ddpm(gr,gt)
#         return fuse


class MAM1(nn.Module):
    def __init__(self, in_channel):
        super(MAM1, self).__init__()
        self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
    def forward(self, x):
        gr = x[0]
        gt = x[1]
        feature_fuse1 = self.fus1(torch.cat([gr, gt], dim=1))
        return feature_fuse1



class CostomAdaptiveAvgPool2D(nn.Module):

    def __init__(self, output_size):

        super(CostomAdaptiveAvgPool2D, self).__init__()
        self.output_size = output_size

    def forward(self, x, H_in, W_in):
        H_out, W_out = [self.output_size, self.output_size] \
            if isinstance(self.output_size, int) \
            else self.output_size

        out_i = []
        for i in range(H_out):
            out_j = []
            for j in range(W_out):
                hs = int(np.floor(i * H_in / H_out))
                he = int(np.ceil((i + 1) * H_in / H_out))

                ws = int(np.floor(j * W_in / W_out))
                we = int(np.ceil((j + 1) * W_in / W_out))

                # print(hs, he, ws, we)
                kernel_size = [he - hs, we - ws]

                out = F.avg_pool2d(x[:, :, hs:he, ws:we], kernel_size)
                out_j.append(out)

            out_j = torch.concat(out_j, -1)
            out_i.append(out_j)

        out_i = torch.concat(out_i, -2)
        return out_i


# class MAM3(nn.Module):
#     def __init__(self, in_channel):
#         super(MAM3, self).__init__()
#         self.channel264 = nn.Sequential(
#             Conv(in_channel*2, in_channel, 3, 1, 1),
#             Conv(in_channel, in_channel//2, 3, 1, 1),
#             Conv(in_channel//2, in_channel//4, 3, 1, 1),
#             Conv(in_channel//4, in_channel//8, 3, 1, 1),
#             Conv(in_channel//8, in_channel//16, 1, 1, 0)
#         )
#         self.avg = CostomAdaptiveAvgPool2D((1, 1))
#         self.xy = nn.Sequential(
#             nn.Conv2d(in_channel//16, 6, 1, 1, 0)
#         )
#         # Start with identity transformation
#         self.xy[-1].weight.data.normal_(mean=0.0, std=5e-4)
#         self.xy[-1].bias.data.zero_()
#         self.fus1 = Conv(in_channel*2, in_channel, 1, 1, 0)
#     def forward(self, x):
#         gr = x[0]
#         gt = x[1]
#         in_ = torch.cat([gr, gt], dim=1)
#         n1 = self.channel264(in_)
#         identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).requires_grad_(False)
#         if in_.is_cuda:
#             identity_theta = identity_theta.cuda().detach()
#         B,C,H,W = n1.size()
#         n1 = self.avg(n1, H, W)
#         deta = self.xy(n1)
#         bsize = deta.shape[0]
#         affine_matrix = deta.view(bsize, -1) + identity_theta.unsqueeze(0).repeat(bsize, 1)
#         # affine_matrix = affine_matrix.half()
#         wrap_grid = F.affine_grid(affine_matrix.view(-1, 2, 3), in_.size(), align_corners=True).permute(0, 3, 1,2).half()
#         wrap_gr = F.grid_sample(gr, wrap_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros',align_corners=True)
#         fuse = self.fus1(torch.cat([wrap_gr, gt], dim=1))
#         return fuse


# class MySTN(nn.Module):
#     def __init__(self, in_ch, mode='Curve'):
#         super(MySTN, self).__init__()
#         self.mode = mode
#         self.down_block_1 = nn.Sequential(
#             convblock(in_ch*2, 128, 3, 2, 1),
#             convblock(128, 128, 1, 1, 0)
#         )
#         self.down_block_2 = nn.Sequential(
#             convblock(128, 128, 3, 2, 1),
#             convblock(128, 128, 1, 1, 0)
#         )
#         self.down_block_3 = nn.Sequential(
#             convblock(128, 128, 3, 2, 1),
#             convblock(128, 128, 1, 1, 0),
#         )
#         self.deta = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(128,6,1,1,0)
#         )
#         # Start with identity transformation
#         self.deta[-1].weight.data.normal_(mean=0.0, std=5e-4)
#         self.deta[-1].bias.data.zero_()
#         self.affine_matrix = None
#         self.wrap_grid = None
#         self.fus1 = Conv(in_ch * 2, in_ch, 1, 1, 0)
#     def forward(self, x):
#         rgb = x[0]
#         t = x[1]
#         in_ = torch.cat([rgb, t], dim=1)
#         size = in_.shape[2:]
#         n1 = self.down_block_1(in_)
#         n2 = self.down_block_2(n1)
#         identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.half).requires_grad_(False)
#         if in_.is_cuda:
#             identity_theta = identity_theta.cuda()
#         n3 = self.down_block_3(n2)
#         deta = self.deta(n3)
#         bsize = deta.shape[0]
#         affine_matrix = deta.view(bsize,-1) + identity_theta.unsqueeze(0).repeat(bsize, 1)
#         wrap_grid = F.affine_grid(affine_matrix.view(-1, 2, 3), in_.size(),align_corners=True).permute(0, 3, 1, 2)
#         wrap_x = F.grid_sample(rgb, wrap_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
#         fuse = self.fus1(torch.cat([wrap_x, t], dim=1))
#         return fuse


class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


# 三个分支concat操作
class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)

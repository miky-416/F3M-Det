# TPAMI 2024：Frequency-aware Feature Fusion for Dense Image Prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.carafe import normal_init, xavier_init, carafe
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    # 生成水平和垂直方向上的Hamming窗
    # hamming_x = np.blackman(M)
    # hamming_x = np.kaiser(M)
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d


class FreqFusion1x(nn.Module):
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,
                 align_corners=False,
                 upsample_mode='nearest',

                 comp_feat_upsample=True,  # use ALPF & AHPF for init upsampling
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.comp_feat_upsample = comp_feat_upsample

        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoderL1 = nn.Conv2d(  # ALPF generator
            self.compressed_channels*2,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.content_encoderL2 = nn.Conv2d(  # ALPF generator
            self.compressed_channels*2,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.content_encoderH1 = nn.Conv2d(  # AHPF generator
            self.compressed_channels*2,
            highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.content_encoderH2 = nn.Conv2d(  # AHPF generator
            self.compressed_channels*2,
            highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        lowpass_pad = 0
        highpass_pad = 0
        self.register_buffer('hamming_lowpass', torch.FloatTensor(
            hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
        self.register_buffer('hamming_highpass', torch.FloatTensor(
            hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])

        self.init_weights()

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        # hr_feat(B,C,H,W), lr_feat(B,C,H,W)
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        compressed_feat = torch.cat((compressed_hr_feat, compressed_lr_feat), dim=1)

        # 从hr_feat得到初始高通滤波特征
        mask_hr_feat = self.content_encoderH1(compressed_feat)
        # kernel归一化得到初始高通滤波
        mask_hr_init = self.kernel_normalizer(mask_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass)
        # 利用初始高通滤波对压缩hr_feat的高频增强 （x-x的低通结果=x的高通结果）
        compressed_hr_feat_s1 = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat.half(),
                                                                                 mask_hr_init.half(),
                                                                                 self.highpass_kernel,
                                                                                 self.up_group, 1)

        mask_lr_feat = self.content_encoderL1(compressed_feat)
        mask_lr_init = self.kernel_normalizer(mask_lr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass)
        compressed_lr_feat_s1 = carafe(compressed_lr_feat.half(), mask_lr_init.half(), self.lowpass_kernel,
                                       self.up_group, 1)

        compressed_feat_s1cat = torch.cat((compressed_hr_feat_s1, compressed_lr_feat_s1), dim=1)

        mask_hr = self.content_encoderH2(compressed_feat_s1cat)
        mask_lr = self.content_encoderL2(compressed_feat_s1cat)


        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        lr_feat_out = carafe(lr_feat.half(), mask_lr.half(), self.lowpass_kernel, self.up_group, 1)


        mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
        hr_feat_out = hr_feat + hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)

        return hr_feat_out + lr_feat_out

    def init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoderL1, std=0.001)
        normal_init(self.content_encoderL2, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoderH1, std=0.001)
            normal_init(self.content_encoderH2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        # print(hamming)
        # print(mask.shape)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask


class FreqFusion2x(nn.Module):
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,
                 align_corners=False,
                 upsample_mode='nearest',

                 comp_feat_upsample=True,  # use ALPF & AHPF for init upsampling
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.comp_feat_upsample = comp_feat_upsample

        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoderL1 = nn.Conv2d(  # ALPF generator
            self.compressed_channels * 2,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.content_encoderL2 = nn.Conv2d(  # ALPF generator
            self.compressed_channels * 2,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.content_encoderH1 = nn.Conv2d(  # AHPF generator
            self.compressed_channels * 2,
            highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.content_encoderH2 = nn.Conv2d(  # AHPF generator
            self.compressed_channels * 2,
            highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        lowpass_pad = 0
        highpass_pad = 0
        self.register_buffer('hamming_lowpass', torch.FloatTensor(
            hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
        self.register_buffer('hamming_highpass', torch.FloatTensor(
            hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])

        self.init_weights()

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        # hr_feat(B,C,2*H,2*W), lr_feat(B,C,H,W)
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        compressed_lr_feat = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:], mode='nearest')
        compressed_feat = torch.cat((compressed_hr_feat, compressed_lr_feat), dim=1)

        # 从hr_feat得到初始高通滤波特征
        mask_hr_feat = self.content_encoderH1(compressed_feat)
        # kernel归一化得到初始高通滤波
        mask_hr_init = self.kernel_normalizer(mask_hr_feat, self.highpass_kernel,
                                              hamming=self.hamming_highpass)
        # 利用初始高通滤波对压缩hr_feat的高频增强 （x-x的低通结果=x的高通结果）
        compressed_hr_feat_s1 = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat.half(),
                                                                                 mask_hr_init.half(),
                                                                                 self.highpass_kernel,
                                                                                 self.up_group, 1)

        mask_lr_feat = self.content_encoderL1(compressed_feat)
        mask_lr_init = self.kernel_normalizer(mask_lr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass)
        compressed_lr_feat_s1 = carafe(compressed_lr_feat.half(), mask_lr_init.half(), self.lowpass_kernel,
                                       self.up_group, 1)

        compressed_feat_s1cat = torch.cat((compressed_hr_feat_s1, compressed_lr_feat_s1), dim=1)

        mask_hr = self.content_encoderH2(compressed_feat_s1cat)
        mask_lr = self.content_encoderL2(compressed_feat_s1cat)



        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        lr_feat_out = carafe(lr_feat.half(), mask_lr.half(), self.lowpass_kernel, self.up_group, 2)

        mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
        hr_feat_out = hr_feat + hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)

        return hr_feat_out + lr_feat_out

    def init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoderL1, std=0.001)
        normal_init(self.content_encoderL2, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoderH1, std=0.001)
            normal_init(self.content_encoderH2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        # print(hamming)
        # print(mask.shape)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask



if __name__ == '__main__':

    hr_feat = torch.rand(1, 128, 512, 512)
    lr_feat = torch.rand(1, 128, 256, 256)
    model = FreqFusion2x(hr_channels=128, lr_channels=128)
    mask_lr, hr_feat, lr_feat = model(hr_feat=hr_feat, lr_feat=lr_feat)
    print(mask_lr.shape)


import torch
import torch.nn as nn


class Fuser(nn.Module):
    """
    Fuse the two input images.
    """

    def __init__(self,  feather_num):
        super().__init__()

        # attention layer
        self.att_a_conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.att_b_conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)

        # dilation conv layer
        self.dil_conv_1 = nn.Sequential(nn.Conv2d(feather_num*2, feather_num, 3, 1, 1, 1), nn.BatchNorm2d(feather_num), nn.ReLU())
        self.dil_conv_2 = nn.Sequential(nn.Conv2d(feather_num*2, feather_num, 3, 1, 2, 2), nn.BatchNorm2d(feather_num), nn.ReLU())
        self.dil_conv_3 = nn.Sequential(nn.Conv2d(feather_num*2, feather_num, 3, 1, 3, 3), nn.BatchNorm2d(feather_num), nn.ReLU())

        # fuse conv layer
        self.fus_conv = nn.Sequential(nn.Conv2d(3 * feather_num, feather_num, 3, padding=1), nn.BatchNorm2d(feather_num//2), nn.Tanh())

    def forward(self, vis, ir):
        """
        :param im_p: image pair
        """

        # attention
        att_a = self._attention(self.att_a_conv, vis)
        att_b = self._attention(self.att_b_conv, ir)

        # focus on attention
        im_a_att = vis * att_a
        im_b_att = ir * att_b

        # image concat
        im_cat = torch.cat([im_a_att, im_b_att], dim=1)

        # dilation
        dil_1 = self.dil_conv_1(im_cat)
        dil_2 = self.dil_conv_2(im_cat)
        dil_3 = self.dil_conv_3(im_cat)

        # feather concat
        f_cat = torch.cat([dil_1, dil_2, dil_3], dim=1)

        # fuse
        im_f_n = self.fus_conv(f_cat)

        return im_f_n

    @staticmethod
    def _attention(att_conv, vis):
        # x = torch.cat([im_x, im_f], dim=1)
        vis_max, _ = torch.max(vis, dim=1, keepdim=True)
        vis_avg = torch.mean(vis, dim=1, keepdim=True)
        vis = torch.cat([vis_max, vis_avg], dim=1)
        vis = att_conv(vis)
        vis_weight = torch.sigmoid(vis)
        return vis_weight
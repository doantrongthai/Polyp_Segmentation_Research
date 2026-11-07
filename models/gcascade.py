"""
G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation.

Reference:
    Rahman, M.M., & Marculescu, R. (2024).
    G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation.
    IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024.
    Repository: https://github.com/SLDGroup/G-CASCADE

Architecture:
    Encoder : PVTv2-B2 (Pyramid Vision Transformer v2)
    Decoder : Cascaded Graph Convolutional Blocks (GCB / Grapher) with relative position embedding,
              Up-Convolution Blocks (UCB), and Spatial Attention (SPA).
    Outputs : [p1, p2, p3, p4] (4 multi-scale maps, p4 at outputs[-1] is finest prediction)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models._backbone.gcn_lib.torch_vertex import Grapher as GCB
from models import register_model


class UCB(nn.Module):
    """Up-convolution block."""

    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1, activation='relu'):
        super(UCB, self).__init__()
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu6':
            self.activation = nn.ReLU6(inplace=True)
        elif activation == 'hardswish':
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=True),
            nn.BatchNorm2d(ch_in),
            self.activation,
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.up(x)


class SPA(nn.Module):
    """Spatial attention block."""

    def __init__(self, kernel_size=7):
        super(SPA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class GCASCADE(nn.Module):
    """Cascaded Graph Convolutional Decoding (GCASCADE)."""

    def __init__(self, channels=None, drop_path_rate=0.0, img_size=352, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu'):
        super(GCASCADE, self).__init__()
        if channels is None:
            channels = [512, 320, 128, 64]
        self.k = k
        self.activation = activation
        self.gcb_act = gcb_act
        self.conv = conv
        self.padding = padding

        self.ucb_ks = 3
        self.ucb_stride = 1
        self.ucb_pad = self.ucb_ks // 2

        self.gcb_norm = 'batch'
        self.bias = True
        self.epsilon = 0.2
        self.use_stochastic = False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1, 1, 4, 2]
        self.dpr = [self.drop_path] * 4
        self.num_knn = [self.k] * 4
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = (img_size // 4) * (img_size // 4)

        self.gcb4 = nn.Sequential(
            GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW // (4 * 4 * 4), drop_path=self.dpr[0],
                relative_pos=True, padding=self.padding),
        )

        self.ucb3 = UCB(ch_in=channels[0], ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(
            GCB(channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW // (4 * 4), drop_path=self.dpr[1],
                relative_pos=True, padding=self.padding),
        )

        self.ucb2 = UCB(ch_in=channels[1], ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(
            GCB(channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW // 4, drop_path=self.dpr[2],
                relative_pos=True, padding=self.padding),
        )

        self.ucb1 = UCB(ch_in=channels[2], ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(
            GCB(channels[3], self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                relative_pos=True, padding=self.padding),
        )

        self.spa = SPA()

    def forward(self, x, skips):
        d4 = self.gcb4(x)
        d4 = self.spa(d4) * d4

        d3 = self.ucb3(d4)
        d3 = d3 + skips[0]
        d3 = self.gcb3(d3)
        d3 = self.spa(d3) * d3

        d2 = self.ucb2(d3)
        d2 = d2 + skips[1]
        d2 = self.gcb2(d2)
        d2 = self.spa(d2) * d2

        d1 = self.ucb1(d2)
        d1 = d1 + skips[2]
        d1 = self.gcb1(d1)
        d1 = self.spa(d1) * d1

        return d4, d3, d2, d1


# ---------------------------------------------------------------------------
# PVT_GCASCADE Model
# ---------------------------------------------------------------------------

@register_model('gcascade')
class GCASCADENet(nn.Module):
    """
    PVT_GCASCADE: Efficient Cascaded Graph Convolutional Decoding (WACV 2024).
    Matches official SLDGroup/G-CASCADE implementation and paper parameters exactly.
    """

    def __init__(
        self,
        n_class: int = 1,
        img_size: int = 352,
        k: int = 11,
        padding: int = 5,
        conv: str = 'mr',
        gcb_act: str = 'gelu',
        activation: str = 'relu',
        pretrained: bool = False
    ):
        super(GCASCADENet, self).__init__()
        self.n_class = n_class

        self.conv_1cto3c = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.backbone = timm.create_model('pvt_v2_b2', pretrained=pretrained, features_only=True)
        self.channels = [512, 320, 128, 64]

        self.decoder = GCASCADE(
            channels=self.channels,
            img_size=img_size,
            k=k,
            padding=padding,
            conv=conv,
            gcb_act=gcb_act,
            activation=activation
        )

        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x: torch.Tensor):
        if x.size(1) == 1:
            x = self.conv_1cto3c(x)

        x1, x2, x3, x4 = self.backbone(x)
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])

        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear', align_corners=False)

        # Multi-scale outputs: p1 (1/32), p2 (1/16), p3 (1/8), p4 (finest, used in fuse_outputs)
        return [p1, p2, p3, p4]

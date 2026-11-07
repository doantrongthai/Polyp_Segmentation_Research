"""
EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation.

Reference:
    Rahman, M.M., Munir, M., & Marculescu, R. (2024).
    EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation.
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
    Repository: https://github.com/SLDGroup/EMCAD

Architecture:
    Encoder : PVTv2-B2 (default, 26.76M params) or PVTv2-B0 (lightweight, 3.92M params)
    Decoder : Multi-Scale Depth-wise Convolutions (MSDC), Multi-Scale Convolution Blocks (MSCB),
              Efficient Up-Convolution Blocks (EUCB), Large-kernel Grouped Attention Gates (LGAG),
              Channel Attention Blocks (CAB), Spatial Attention Blocks (SAB).
    Outputs : [p4, p3, p2, p1] (4 multi-scale supervision maps, p1 at outputs[-1] is finest prediction)
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_tf_
from models import register_model


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'relu6':
        return nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'hswish':
        return nn.Hardswish(inplace)
    else:
        raise NotImplementedError(f'Activation layer [{act}] is not supported')


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class MSDC(nn.Module):
    """Multi-scale depth-wise convolution."""

    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    """Multi-scale convolution block."""

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=None, expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        super(MSCB, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = (self.stride == 1)

        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = sum(msdc_outs)
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        return out


def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=None, expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
    if kernel_sizes is None:
        kernel_sizes = [1, 3, 5]
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(mscb)
    if n > 1:
        for _ in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(mscb)
    return nn.Sequential(*convs)


class EUCB(nn.Module):
    """Efficient up-convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


class LGAG(nn.Module):
    """Large-kernel grouped attention gate."""

    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class CAB(nn.Module):
    """Channel attention block."""

    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SAB(nn.Module):
    """Spatial attention block."""

    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()
        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class EMCAD(nn.Module):
    """Efficient multi-scale convolutional attention decoding (EMCAD)."""

    def __init__(self, channels=None, kernel_sizes=None, expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(EMCAD, self).__init__()
        if channels is None:
            channels = [512, 320, 128, 64]
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]
        eucb_ks = 3

        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag3 = LGAG(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=lgag_ks, groups=channels[1] // 2)
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag2 = LGAG(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=lgag_ks, groups=channels[2] // 2)
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgag1 = LGAG(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2), kernel_size=lgag_ks, groups=int(channels[3] / 2))
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        self.sab = SAB()

    def forward(self, x, skips):
        d4 = self.cab4(x) * x
        d4 = self.sab(d4) * d4
        d4 = self.mscb4(d4)

        d3 = self.eucb3(d4)
        x3 = self.lgag3(g=d3, x=skips[0])
        d3 = d3 + x3
        d3 = self.cab3(d3) * d3
        d3 = self.sab(d3) * d3
        d3 = self.mscb3(d3)

        d2 = self.eucb2(d3)
        x2 = self.lgag2(g=d2, x=skips[1])
        d2 = d2 + x2
        d2 = self.cab2(d2) * d2
        d2 = self.sab(d2) * d2
        d2 = self.mscb2(d2)

        d1 = self.eucb1(d2)
        x1 = self.lgag1(g=d1, x=skips[2])
        d1 = d1 + x1
        d1 = self.cab1(d1) * d1
        d1 = self.sab(d1) * d1
        d1 = self.mscb1(d1)

        return [d4, d3, d2, d1]


# ---------------------------------------------------------------------------
# EMCADNet Model
# ---------------------------------------------------------------------------

@register_model('emcad')
class EMCADNet(nn.Module):
    """
    EMCADNet: Efficient Multi-scale Convolutional Attention Decoding Network (CVPR 2024).
    Matches official SLDGroup/EMCAD implementation and paper parameters exactly.
    """

    def __init__(
        self,
        num_classes: int = 1,
        kernel_sizes=None,
        expansion_factor: int = 2,
        dw_parallel: bool = True,
        add: bool = True,
        lgag_ks: int = 3,
        activation: str = 'relu',
        encoder: str = 'pvt_v2_b2',
        pretrained: bool = False
    ):
        super(EMCADNet, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        if encoder in ('pvt_v2_b2', 'pvt_v2_b1', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5'):
            self.backbone = timm.create_model(encoder, pretrained=pretrained, features_only=True)
            channels = [512, 320, 128, 64]
        elif encoder == 'pvt_v2_b0':
            self.backbone = timm.create_model(encoder, pretrained=pretrained, features_only=True)
            channels = [256, 160, 64, 32]
        else:
            self.backbone = timm.create_model('pvt_v2_b2', pretrained=pretrained, features_only=True)
            channels = [512, 320, 128, 64]

        self.decoder = EMCAD(
            channels=channels,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation
        )

        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)

    def forward(self, x: torch.Tensor):
        if x.size(1) == 1:
            x = self.conv(x)

        x1, x2, x3, x4 = self.backbone(x)
        dec_outs = self.decoder(x4, [x3, x2, x1])

        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear', align_corners=False)
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear', align_corners=False)

        # Multi-scale outputs: p4 (1/32), p3 (1/16), p2 (1/8), p1 (finest, used in fuse_outputs)
        return [p4, p3, p2, p1]

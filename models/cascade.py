"""
CASCADE: Context-Aware Synergistic Cascaded Network for Medical Image Segmentation.

Reference:
    Rahman, M. M., & Marculescu, R. (2023).
    CASCADE: Context-Aware Synergistic Cascaded Network for Medical Image Segmentation.
    IEEE WACV 2023, pp. 2488-2497.
    Official Repository: https://github.com/SLDGroup/CASCADE

Architecture:
    - Backbone: PVTv2-B2 Transformer Encoder
    - Decoder: Multi-stage CASCADE decoder with Channel Attention (CAM), Spatial Attention (SA),
      and Attention Gate (AG) skip connections
    - Deep supervision across all 4 multi-scale decoding stages: p1 (1/32), p2 (1/16), p3 (1/8), p4 (1/4)
    - All 4 maps upsampled to input resolution (352x352) with p4 as the finest primary prediction
    - Parameters: ~35.27M (35,272,795)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._backbone.pvt_v2 import pvt_v2_b2
from models import register_model


class conv_block(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class Attention_block(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CASCADEDecoder(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64]):
        super(CASCADEDecoder, self).__init__()

        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2 * channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2 * channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=32)
        self.ConvBlock1 = conv_block(ch_in=2 * channels[3], ch_out=channels[3])

        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2 * channels[1])
        self.CA2 = ChannelAttention(2 * channels[2])
        self.CA1 = ChannelAttention(2 * channels[3])

        self.SA = SpatialAttention()

    def forward(self, x, skips):
        d4 = self.Conv_1x1(x)

        # CAM4
        d4 = self.CA4(d4) * d4
        d4 = self.SA(d4) * d4
        d4 = self.ConvBlock4(d4)

        # upconv3
        d3 = self.Up3(d4)
        x3 = self.AG3(g=d3, x=skips[0])
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.CA3(d3) * d3
        d3 = self.SA(d3) * d3
        d3 = self.ConvBlock3(d3)

        # upconv2
        d2 = self.Up2(d3)
        x2 = self.AG2(g=d2, x=skips[1])
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.CA2(d2) * d2
        d2 = self.SA(d2) * d2
        d2 = self.ConvBlock2(d2)

        # upconv1
        d1 = self.Up1(d2)
        x1 = self.AG1(g=d1, x=skips[2])
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.CA1(d1) * d1
        d1 = self.SA(d1) * d1
        d1 = self.ConvBlock1(d1)

        return d4, d3, d2, d1


@register_model("cascade")
@register_model("pvt_cascade")
class CASCADE(nn.Module):
    """
    PVT-CASCADE Architecture (Rahman & Marculescu, WACV 2023)
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = False, in_channels: int = 3, **kwargs):
        super(CASCADE, self).__init__()

        # Conv block to handle optional 1-channel input
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # PVTv2-B2 backbone (num_classes=0 matches official head-less configuration)
        self.backbone = pvt_v2_b2(pretrained=pretrained, num_classes=0)

        # Decoder
        self.decoder = CASCADEDecoder(channels=[512, 320, 128, 64])

        # Prediction heads
        self.out_head1 = nn.Conv2d(512, num_classes, 1)
        self.out_head2 = nn.Conv2d(320, num_classes, 1)
        self.out_head3 = nn.Conv2d(128, num_classes, 1)
        self.out_head4 = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = self.conv(x)

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

        # (p1, p2, p3, p4) ordered coarse-to-fine so finest primary prediction is at [-1]
        return p1, p2, p3, p4

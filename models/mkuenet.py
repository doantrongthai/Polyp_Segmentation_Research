"""
MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation.

Reference:
    Rahman, M.M., & Marculescu, R. (2025).
    MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation.
    ICCV 2025 CVAMD Workshop.
    https://arxiv.org/abs/2509.18493
    https://github.com/SLDGroup/MK-UNet

Architecture:
    Ultra-lightweight U-Net with Multi-Kernel Depth-wise Conv Block (MKDC).
    MKDC: parallel depthwise convolutions at multiple kernel sizes (1,3,5,7)
          + Channel Attention + Spatial Attention + Grouped Gated Attention.
    Params: ~0.316M, FLOPs: ~0.314G

Note: Official code not yet released at time of implementation (Sept 2025).
Architecture reconstructed from paper description (arXiv:2509.18493).

Registered as: 'mkuenet'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


# ---------------------------------------------------------------------------
# MKDC Block (Multi-Kernel Depth-wise Convolution)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super().__init__()
        mid = max(1, in_ch // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_ch, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.fc(self.avg(x))
        m = self.fc(self.max(x))
        return x * self.sigmoid(a + m)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        mask = self.conv(torch.cat([avg, mx], dim=1))
        return x * self.sigmoid(mask)


class GroupedGatedAttention(nn.Module):
    """Grouped Gated Attention: split channels into groups, gate each group."""

    def __init__(self, in_ch, groups=4):
        super().__init__()
        # Ensure groups divides in_ch
        while groups > 1 and in_ch % groups != 0:
            groups -= 1
        self.groups = groups
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, groups=groups, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class MKDCBlock(nn.Module):
    """
    Multi-Kernel Depth-wise Convolution Block.
    Parallel DW convs at k=[1,3,5,7] + pointwise + triple attention.
    """

    def __init__(self, in_ch, out_ch, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, k, padding=k // 2, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.GELU(),
            ) for k in kernels
        ])
        # Fuse multi-kernel features
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch * len(kernels), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        # Attention
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()
        self.gga = GroupedGatedAttention(out_ch)

        # Residual projection if channels differ
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        feats = torch.cat([conv(x) for conv in self.dw_convs], dim=1)
        out = self.pw(feats)
        out = self.ca(out)
        out = self.sa(out)
        out = self.gga(out)
        return out + residual


# ---------------------------------------------------------------------------
# MK-UNet
# ---------------------------------------------------------------------------

@register_model('mkuenet')
class MKUNet(nn.Module):
    """
    MK-UNet: Ultra-lightweight U-Net with MKDC blocks.

    ~0.316M parameters, 0.314 GFLOPs.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 10):
        super().__init__()
        # Encoder
        self.enc1 = MKDCBlock(in_ch, base)
        self.enc2 = MKDCBlock(base, base * 2)
        self.enc3 = MKDCBlock(base * 2, base * 4)
        self.enc4 = MKDCBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = MKDCBlock(base * 8, base * 16)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = MKDCBlock(base * 16 + base * 8, base * 8)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = MKDCBlock(base * 8 + base * 4, base * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = MKDCBlock(base * 4 + base * 2, base * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = MKDCBlock(base * 2 + base, base)

        # Output
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)

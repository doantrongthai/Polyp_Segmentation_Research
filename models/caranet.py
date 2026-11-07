"""
CaraNet: Context Axial Reverse Attention Network for Small Medical Objects Segmentation.

Reference:
    Lou, A., Guan, S., & Loew, M. (2022).
    CaraNet: Context axial reverse attention network for small medical objects segmentation.
    Medical Image Analysis, 102391.
    Official Repository: https://github.com/AngeLouCN/CaraNet

Architecture:
    - Backbone: Res2Net-101 v1b (ImageNet pretrained)
    - CFPModule: Context Feature Pyramid Module
    - AA_kernel: Axial Attention Kernel (H-attention and W-attention)
    - aggregation: Dense Partial Decoder
    - Reverse Attention branches with Context Axial Attention
    - Outputs: lateral_map_1 (coarse), lateral_map_2, lateral_map_3, lateral_map_5 (fine)

Interface adaptation:
    - Registered as 'caranet' in model registry.
    - Output format aligned with fuse_outputs / split_model_output convention:
      (lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_5), where lateral_map_5
      is the primary finest prediction at index -1.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models._backbone.res2net import res2net101_v1b_26w_4s


# ---------------------------------------------------------------------------
# Building Blocks (Preserved from official AngeLouCN/CaraNet)
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Conv2d with optional BatchNorm + PReLU."""

    def __init__(
        self,
        nIn: int,
        nOut: int,
        kSize,
        stride: int = 1,
        padding: int = 0,
        dilation=(1, 1),
        groups: int = 1,
        bn_acti: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            kernel_size=kSize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.bn_acti:
            out = self.bn_relu(out)
        return out


class BNPReLU(nn.Module):
    """BatchNorm2d + PReLU activation."""

    def __init__(self, nIn: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.acti(self.bn(x))


class self_attn(nn.Module):
    """Axial self-attention along horizontal or vertical axis."""

    def __init__(self, in_channels: int, mode: str = "hw"):
        super().__init__()
        self.mode = mode
        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1), stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = x.size()
        axis = 1
        if "h" in self.mode:
            axis *= height
        if "w" in self.mode:
            axis *= width

        view = (batch_size, -1, axis)
        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)
        out = self.gamma * out + x
        return out


class AA_kernel(nn.Module):
    """Axial Attention kernel combining H and W attention."""

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1, stride=1, padding=0)
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3), stride=1, padding=1)
        self.Hattn = self_attn(out_channel, mode="h")
        self.Wattn = self_attn(out_channel, mode="w")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)
        return Wx


class CFPModule(nn.Module):
    """Context Feature Pyramid Module."""

    def __init__(self, nIn: int, d: int = 1, KSize: int = 3, dkSize: int = 3):
        super().__init__()
        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)

        self.dconv_4_1 = Conv(
            nIn // 4, nIn // 16, (dkSize, dkSize), 1,
            padding=(1 * d + 1, 1 * d + 1), dilation=(d + 1, d + 1),
            groups=nIn // 16, bn_acti=True,
        )
        self.dconv_4_2 = Conv(
            nIn // 16, nIn // 16, (dkSize, dkSize), 1,
            padding=(1 * d + 1, 1 * d + 1), dilation=(d + 1, d + 1),
            groups=nIn // 16, bn_acti=True,
        )
        self.dconv_4_3 = Conv(
            nIn // 16, nIn // 8, (dkSize, dkSize), 1,
            padding=(1 * d + 1, 1 * d + 1), dilation=(d + 1, d + 1),
            groups=nIn // 16, bn_acti=True,
        )

        self.dconv_1_1 = Conv(
            nIn // 4, nIn // 16, (dkSize, dkSize), 1,
            padding=(1, 1), dilation=(1, 1),
            groups=nIn // 16, bn_acti=True,
        )
        self.dconv_1_2 = Conv(
            nIn // 16, nIn // 16, (dkSize, dkSize), 1,
            padding=(1, 1), dilation=(1, 1),
            groups=nIn // 16, bn_acti=True,
        )
        self.dconv_1_3 = Conv(
            nIn // 16, nIn // 8, (dkSize, dkSize), 1,
            padding=(1, 1), dilation=(1, 1),
            groups=nIn // 16, bn_acti=True,
        )

        self.dconv_2_1 = Conv(
            nIn // 4, nIn // 16, (dkSize, dkSize), 1,
            padding=(int(d / 4 + 1), int(d / 4 + 1)), dilation=(int(d / 4 + 1), int(d / 4 + 1)),
            groups=nIn // 16, bn_acti=True,
        )
        self.dconv_2_2 = Conv(
            nIn // 16, nIn // 16, (dkSize, dkSize), 1,
            padding=(int(d / 4 + 1), int(d / 4 + 1)), dilation=(int(d / 4 + 1), int(d / 4 + 1)),
            groups=nIn // 16, bn_acti=True,
        )
        self.dconv_2_3 = Conv(
            nIn // 16, nIn // 8, (dkSize, dkSize), 1,
            padding=(int(d / 4 + 1), int(d / 4 + 1)), dilation=(int(d / 4 + 1), int(d / 4 + 1)),
            groups=nIn // 16, bn_acti=True,
        )

        self.dconv_3_1 = Conv(
            nIn // 4, nIn // 16, (dkSize, dkSize), 1,
            padding=(int(d / 2 + 1), int(d / 2 + 1)), dilation=(int(d / 2 + 1), int(d / 2 + 1)),
            groups=nIn // 16, bn_acti=True,
        )
        self.dconv_3_2 = Conv(
            nIn // 16, nIn // 16, (dkSize, dkSize), 1,
            padding=(int(d / 2 + 1), int(d / 2 + 1)), dilation=(int(d / 2 + 1), int(d / 2 + 1)),
            groups=nIn // 16, bn_acti=True,
        )
        self.dconv_3_3 = Conv(
            nIn // 16, nIn // 8, (dkSize, dkSize), 1,
            padding=(int(d / 2 + 1), int(d / 2 + 1)), dilation=(int(d / 2 + 1), int(d / 2 + 1)),
            groups=nIn // 16, bn_acti=True,
        )

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)

        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)

        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)

        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)
        output_4 = torch.cat([o4_1, o4_2, o4_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1, ad2, ad3, ad4], 1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


class aggregation(nn.Module):
    """Dense aggregation module for partial decoding."""

    def __init__(self, channel: int = 32):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_upsample1 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample2 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample3 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample4 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample5 = Conv(2 * 32, 2 * 32, 3, 1, padding=1)

        self.conv_concat2 = Conv(2 * 32, 2 * 32, 3, 1, padding=1)
        self.conv_concat3 = Conv(3 * 32, 3 * 32, 3, 1, padding=1)
        self.conv4 = Conv(3 * 32, 3 * 32, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(3 * 32, 1, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


# ---------------------------------------------------------------------------
# CaraNet Main Architecture
# ---------------------------------------------------------------------------

@register_model("caranet")
class CaraNet(nn.Module):
    """Context Axial Reverse Attention Network (CaraNet)."""

    def __init__(self, channel: int = 32, pretrained: bool = True):
        super().__init__()
        # Backbone: Res2Net101-v1b
        self.resnet = res2net101_v1b_26w_4s(pretrained=pretrained)

        # Receptive Field Block channels (same module names as upstream for state_dict 1:1 match)
        self.rfb2_1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1 = Conv(1024, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1 = Conv(2048, 32, 3, 1, padding=1, bn_acti=True)

        # Partial Decoder
        self.agg1 = aggregation(channel)

        self.CFP_1 = CFPModule(32, d=8)
        self.CFP_2 = CFPModule(32, d=8)
        self.CFP_3 = CFPModule(32, d=8)

        self.ra1_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.aa_kernel_1 = AA_kernel(32, 32)
        self.aa_kernel_2 = AA_kernel(32, 32)
        self.aa_kernel_3 = AA_kernel(32, 32)

    def forward(self, x: torch.Tensor):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode="bilinear", align_corners=False)

        # Attention block 3
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode="bilinear", align_corners=False)
        cfp_out_1 = self.CFP_3(x4_rfb)
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)

        ra_3 = self.ra3_conv1(aa_atten_3_o)
        ra_3 = self.ra3_conv2(ra_3)
        ra_3 = self.ra3_conv3(ra_3)

        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode="bilinear", align_corners=False)

        # Attention block 2
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode="bilinear", align_corners=False)
        cfp_out_2 = self.CFP_2(x3_rfb)
        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)

        ra_2 = self.ra2_conv1(aa_atten_2_o)
        ra_2 = self.ra2_conv2(ra_2)
        ra_2 = self.ra2_conv3(ra_2)

        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode="bilinear", align_corners=False)

        # Attention block 1
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode="bilinear", align_corners=False)
        cfp_out_3 = self.CFP_1(x2_rfb)
        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)

        ra_1 = self.ra1_conv1(aa_atten_1_o)
        ra_1 = self.ra1_conv2(ra_1)
        ra_1 = self.ra1_conv3(ra_1)

        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode="bilinear", align_corners=False)

        # Order: coarse -> fine, so lateral_map_5 (primary prediction) is at [-1]
        return lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_5

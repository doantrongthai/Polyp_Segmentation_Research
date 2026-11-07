"""
DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation.

Reference:
    Jha, D., Smedsrud, P. H., Riegler, M. A., Johansen, D., De Lange, T., Halvorsen, P.,
    & Johansen, H. D. (2020).
    DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation.
    IEEE CBMS 2020, pp. 558-563.
    Official Repository: https://github.com/DebeshJha/2020-CBMS-DoubleU-Net

Architecture:
    - First U-Net: VGG-19 encoder + ASPP + Modified Decoder 1
    - Second U-Net: Squeeze-and-Excitation CNN encoder + ASPP + Modified Decoder 2
    - Residual multiplication bridge between stages
    - Parameters: ~29.29M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

try:
    from torchvision.models import VGG19_Weights
    HAS_VGG_WEIGHTS = True
except ImportError:
    HAS_VGG_WEIGHTS = False

from models import register_model


class Conv2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, padding: int = 1,
                 dilation: int = 1, bias: bool = False, act: bool = True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )
        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)
        self.c5 = Conv2D(out_c * 5, out_c, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x0, x1, x2, x3, x4], dim=1)
        return self.c5(xc)


class conv_block(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.a1 = squeeze_excitation_block(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.a1(x)
        return x


class encoder1(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained:
            if HAS_VGG_WEIGHTS:
                network = vgg19(weights=VGG19_Weights.DEFAULT)
            else:
                network = vgg19(pretrained=True)
        else:
            if HAS_VGG_WEIGHTS:
                network = vgg19(weights=None)
            else:
                network = vgg19(pretrained=False)

        self.x1 = network.features[:4]
        self.x2 = network.features[4:9]
        self.x3 = network.features[9:18]
        self.x4 = network.features[18:27]
        self.x5 = network.features[27:36]

    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        return x5, [x4, x3, x2, x1]


class decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(64 + 512, 256)
        self.c2 = conv_block(512, 128)
        self.c3 = conv_block(256, 64)
        self.c4 = conv_block(128, 32)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip
        x = self.up(x)
        x = torch.cat([x, s1], dim=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], dim=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s3], dim=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, s4], dim=1)
        x = self.c4(x)
        return x


class encoder2(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.pool = nn.MaxPool2d((2, 2))
        self.c1 = conv_block(in_channels, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x1 = self.c1(x)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)
        return p4, [x4, x3, x2, x1]


class decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(832, 256)
        self.c2 = conv_block(640, 128)
        self.c3 = conv_block(320, 64)
        self.c4 = conv_block(160, 32)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], dim=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], dim=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], dim=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], dim=1)
        x = self.c4(x)
        return x


@register_model("double_unet")
@register_model("doubleunet")
class DoubleUNet(nn.Module):
    """
    DoubleU-Net Architecture (Jha et al., CBMS 2020)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, pretrained: bool = False, **kwargs):
        super().__init__()
        self.e1 = encoder1(pretrained=pretrained)
        self.a1 = ASPP(512, 64)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(32, num_classes, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.e2 = encoder2(in_channels=in_channels)
        self.a2 = ASPP(256, 64)
        self.d2 = decoder2()
        self.y2 = nn.Conv2d(32, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x0 = x
        x, skip1 = self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)

        input_x = x0 * self.sigmoid(y1)
        x, skip2 = self.e2(input_x)
        x = self.a2(x)
        x = self.d2(x, skip1, skip2)
        y2 = self.y2(x)

        return y1, y2

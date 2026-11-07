"""
FCBFormer: Fully Convolutional Branch with Transformer for Polyp Segmentation.

Reference:
    Sandoval-Perez, E. et al. (2022).
    FCN-Transformer Feature Fusion for Polyp Segmentation.
    MIUA 2022 / Medical Image Analysis.
    Official Repository: https://github.com/ESandML/FCBFormer

Architecture:
    - TB (Transformer Branch): PVTv2-B3 encoder extracting multi-scale pyramid tokens,
      with Layer Enhancement (LE) and Stepwise Feature Aggregation (SFA).
    - FCB (Fully Convolutional Branch): Multi-resolution residual blocks across 6 scales
      preserving spatial resolution and fine boundary details.
    - PH (Prediction Head): Fusion head merging TB and FCB features to output 1 segmentation map.
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models._backbone.pvt_v2 import pvt_v2_b3


class RB(nn.Module):
    """Residual Block with GroupNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class FCB(nn.Module):
    """Fully Convolutional Branch operating across multiple downsampling levels."""

    def __init__(
        self,
        in_channels: int = 3,
        min_level_channels: int = 32,
        min_channel_mults: list = [1, 1, 2, 2, 4, 4],
        n_levels_down: int = 6,
        n_levels_up: int = 6,
        n_RBs: int = 2,
        in_resolution: int = 352,
    ):
        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h


class TB(nn.Module):
    """Transformer Branch built on PVTv2-B3 encoder with SFA."""

    def __init__(self, pretrained: bool = False):
        super().__init__()

        backbone = pvt_v2_b3(pretrained=pretrained)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([64, 128, 320, 512][i], 64), RB(64, 64), nn.Upsample(size=88)
                )
            )

        self.SFA = nn.ModuleList([])
        for i in range(3):
            self.SFA.append(nn.Sequential(RB(128, 64), RB(64, 64)))

    def get_pyramid(self, x: torch.Tensor):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)
        return pyramid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pyramid = self.get_pyramid(x)
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l


@register_model("fcbformer")
class FCBFormer(nn.Module):
    """FCBFormer: Combined Fully Convolutional Branch and Transformer for Polyp Segmentation."""

    def __init__(self, size: int = 352, pretrained: bool = False):
        super().__init__()
        self.TB = TB(pretrained=pretrained)
        self.FCB = FCB(in_resolution=size)
        self.PH = nn.Sequential(
            RB(64 + 32, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.TB(x)
        x2 = self.FCB(x)
        x1 = self.up_tosize(x1)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)
        return out

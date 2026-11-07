"""
SANet: Shallow Attention Network for Polyp Segmentation.

Reference:
    Wei, J., Hu, Y., Zhang, R., Li, Z., Zhou, S. K., & Cui, S. (2021).
    Shallow Attention Network for Polyp Segmentation.
    MICCAI 2021, LNCS 12901, pp. 699-708.
    Repository: https://github.com/weijun88/SANet

Backbone: Res2Net-50 v1b (26w_4s)
Key Innovation: Shallow Attention mechanism that leverages shallow features to filter background noise
and enhance fuzzy polyp boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models._backbone.res2net import res2net50_v1b_26w_4s
from models import register_model


@register_model('sanet')
class SANet(nn.Module):
    """Shallow Attention Network (MICCAI 2021)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.bkbone = res2net50_v1b_26w_4s(pretrained=pretrained)

        # Projections to 64 channels matching official weijun88/SANet exactly
        self.linear5 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.linear4 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.linear3 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final prediction head
        self.predict = nn.Conv2d(64 * 3, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone returns [x1 (256), x2 (512), x3 (1024), x4 (2048)]
        out2, out3, out4, out5 = self.bkbone(x)

        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)

        # Align to resolution of out3 (44x44 for 352x352 input)
        target_size = out3.size()[2:]
        out5 = F.interpolate(out5, size=target_size, mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=target_size, mode='bilinear', align_corners=True)

        # Progressive shallow attention interaction
        pred = torch.cat([out5, out4 * out5, out3 * out4 * out5], dim=1)
        pred = self.predict(pred)

        # Upsample to match input resolution (352x352)
        return F.interpolate(pred, size=x.shape[2:], mode='bilinear', align_corners=True)

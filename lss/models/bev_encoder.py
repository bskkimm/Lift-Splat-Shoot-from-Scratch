"""ResNet-block bird's-eye-view segmentation decoder."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class BEVEncoder(nn.Module):
    """Map pooled BEV features [B,C,X,Y] to raw logits [B,K,X,Y]."""

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("BEV encoder channel counts must be positive")
        self.in_channels = in_channels
        trunk = resnet18(weights=None, zero_init_residual=True)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            trunk.bn1,
            trunk.relu,
        )
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.fuse = nn.Sequential(
            nn.Conv2d(64 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1),
        )

    def forward(self, bev):
        if bev.ndim != 4 or bev.shape[1] != self.in_channels:
            raise ValueError("BEV features must have shape [B,in_channels,X,Y]")
        height, width = bev.shape[-2:]
        if height < 8 or width < 8 or height % 8 or width % 8:
            raise ValueError("BEV spatial dimensions must be positive and divisible by 8")
        skip = self.layer1(self.stem(bev))
        x = self.layer3(self.layer2(skip))
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = self.fuse(torch.cat((skip, x), dim=1))
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=True)
        return self.output(x)

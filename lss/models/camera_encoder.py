"""EfficientNet-B0 image path for an LSS-style camera encoder."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class CameraEncoder(nn.Module):
    """Predict a latent depth distribution and context for each image pixel.

    Images have shape [B, N, 3, H, W]; H and W must be divisible by 32.
    Outputs are depth probabilities [B, N, D, H/16, W/16] and context
    [B, N, C, H/16, W/16]. Pretrained TorchVision weights are opt-in and
    may require a network download; they are not the original LSS checkpoint.
    """

    def __init__(self, depth_bins, context_channels, pretrained=False):
        super().__init__()
        if depth_bins <= 0 or context_channels <= 0:
            raise ValueError("depth_bins and context_channels must be positive")
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = efficientnet_b0(weights=weights).features[:8]
        self.depth_bins = depth_bins
        self.context_channels = context_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(320 + 112, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Conv2d(512, depth_bins + context_channels, kernel_size=1)

    def forward(self, images):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,N,3,H,W]")
        batch, cameras, channels, height, width = images.shape
        if height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError("image height and width must be positive and divisible by 32")

        x = images.reshape(batch * cameras, channels, height, width)
        for stage in self.features[:6]:
            x = stage(x)
        skip = x
        for stage in self.features[6:8]:
            x = stage(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = self.projection(self.fuse(torch.cat((skip, x), dim=1)))
        depth = x[:, :self.depth_bins].softmax(dim=1)
        context = x[:, self.depth_bins:]
        return (depth.reshape(batch, cameras, self.depth_bins, *depth.shape[-2:]),
                context.reshape(batch, cameras, self.context_channels, *context.shape[-2:]))

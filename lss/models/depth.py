import torch.nn as nn


class DepthContextNet(nn.Module):
    def __init__(self, in_channels=64, depth_bins=80, context_channels=64):
        super().__init__()
        self.depth = nn.Conv2d(in_channels, depth_bins, 1)
        self.context = nn.Conv2d(in_channels, context_channels, 1)

    def forward(self, features):
        b, c, ch, h, w = features.shape
        x = features.reshape(b * c, ch, h, w)
        return self.depth(x).softmax(1).reshape(b, c, -1, h, w), self.context(x).reshape(b, c, -1, h, w)

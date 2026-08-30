import torch.nn as nn


class ImageBackbone(nn.Module):
    def __init__(self, in_channels=3, channels=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, channels, 3, 2, 1), nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 3, 2, 1), nn.ReLU())

    def forward(self, images):
        b, c, _, h, w = images.shape
        return self.net(images.reshape(b * c, 3, h, w)).reshape(b, c, -1, h // 4, w // 4)

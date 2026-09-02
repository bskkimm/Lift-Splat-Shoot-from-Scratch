import torch.nn as nn


class ImageBackbone(nn.Module):
    def __init__(self, in_channels=3, channels=64, name="tiny", pretrained=False):
        super().__init__()
        if name == "tiny":
            self.net = nn.Sequential(nn.Conv2d(in_channels, channels, 3, 2, 1), nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 3, 2, 1), nn.ReLU())
        else:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            self.net = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, nn.Conv2d(512, channels, 1))

    def forward(self, images):
        b, c, _, h, w = images.shape
        output = self.net(images.reshape(b * c, 3, h, w))
        return output.reshape(b, c, output.shape[1], output.shape[2], output.shape[3])


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__(); self.proj = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, feature):
        if isinstance(feature, (list, tuple)): return [self.proj(item) for item in feature]
        return self.proj(feature)

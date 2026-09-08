import torch.nn as nn
from .lss import LSS
from .head import BEVHead


class LSSDetector(nn.Module):
    def __init__(self, classes=10, **kwargs):
        super().__init__(); self.lss = LSS(**kwargs); self.head = BEVHead(classes=classes)
    def forward(self, images, intrinsics, extrinsics, depths):
        bev, depth = self.lss(images, intrinsics, extrinsics, depths)
        output = self.head(bev); output["depth"] = depth; return output

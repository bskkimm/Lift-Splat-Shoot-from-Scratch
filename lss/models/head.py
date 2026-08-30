import torch.nn as nn


class BEVHead(nn.Module):
    def __init__(self, channels=64, classes=10, box_dim=9):
        super().__init__(); self.cls = nn.Conv2d(channels, classes, 1); self.box = nn.Conv2d(channels, box_dim, 1)
    def forward(self, bev): return {"logits": self.cls(bev), "boxes": self.box(bev)}

import torch.nn as nn
import torch


class BEVHead(nn.Module):
    def __init__(self, channels=64, classes=10, box_dim=9):
        super().__init__(); self.cls = nn.Conv2d(channels, classes, 1); self.box = nn.Conv2d(channels, box_dim, 1)
    def forward(self, bev): return {"logits": self.cls(bev), "boxes": self.box(bev)}

    @staticmethod
    def decode(outputs, score_threshold=0.05, max_boxes=100):
        scores = outputs["logits"].sigmoid().flatten(2)
        values, indices = scores.flatten(1).topk(min(max_boxes, scores.shape[1] * scores.shape[2]), dim=1)
        labels = indices // scores.shape[2]; locations = indices % scores.shape[2]
        boxes = outputs["boxes"].flatten(2).gather(2, locations[:, None].expand(-1, outputs["boxes"].shape[1], -1)).transpose(1, 2)
        return [{"scores": v[v >= score_threshold], "labels": l[v >= score_threshold], "boxes": b[v >= score_threshold]} for v, l, b in zip(values, labels, boxes)]

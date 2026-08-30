import torch
from lss.models.head import BEVHead
def test_head_outputs_logits_and_boxes():
    out = BEVHead(classes=3)(torch.randn(2,64,4,5)); assert out["logits"].shape == (2,3,4,5); assert out["boxes"].shape == (2,9,4,5)

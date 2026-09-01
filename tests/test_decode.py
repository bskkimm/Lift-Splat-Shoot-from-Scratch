import torch
from lss.models.head import BEVHead
def test_head_decode_filters_low_scores():
    out = BEVHead(classes=2)(torch.zeros(1,64,2,2)); decoded = BEVHead.decode(out, .5); assert len(decoded) == 1 and decoded[0]["boxes"].shape[1] == 9

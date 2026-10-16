import torch
from lss.models.head import BEVHead
def test_decode_handles_zero_max_boxes():
    out=BEVHead()(torch.zeros(1,64,2,2)); assert BEVHead.decode(out,max_boxes=0)[0]["boxes"].shape[0] == 0

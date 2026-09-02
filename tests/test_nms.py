import torch
from lss.models.head import BEVHead
def test_nms_keeps_highest_score():
    keep = BEVHead.nms(torch.tensor([[1.,1.,1.],[1.,1.,1.]]), torch.tensor([.9,.1])); assert keep.tolist() == [0]

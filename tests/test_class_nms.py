import torch
from lss.models.head import BEVHead
def test_class_aware_nms_keeps_overlapping_different_classes():
    keep=BEVHead.class_aware_nms(torch.tensor([[1.,1.,1.],[1.,1.,1.]]),torch.tensor([.9,.8]),torch.tensor([0,1])); assert keep.numel()==2

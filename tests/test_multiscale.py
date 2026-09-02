import torch
from lss.models.backbone import FPN
def test_fpn_accepts_multiple_feature_levels():
    result = FPN(8, 4)([torch.randn(1,8,4,4), torch.randn(1,8,2,2)])
    assert [item.shape[-2:] for item in result] == [(4,4),(2,2)]

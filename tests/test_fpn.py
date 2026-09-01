import torch
from lss.models.backbone import FPN
def test_fpn_projects_features():
    assert FPN(8, 4)(torch.randn(1,8,3,3)).shape == (1,4,3,3)

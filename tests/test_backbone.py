import torch
from lss.models.backbone import ImageBackbone


def test_backbone_preserves_camera_axis():
    assert ImageBackbone()(torch.randn(2, 3, 3, 16, 20)).shape == (2, 3, 64, 4, 5)

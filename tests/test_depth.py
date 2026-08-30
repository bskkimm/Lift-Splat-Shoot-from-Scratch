import torch
from lss.models.depth import DepthContextNet


def test_depth_context_outputs_normalized_depth():
    depth, context = DepthContextNet(in_channels=16, depth_bins=4, context_channels=8)(torch.randn(2, 3, 16, 4, 5))
    assert depth.shape == (2, 3, 4, 4, 5)
    assert context.shape == (2, 3, 8, 4, 5)
    assert torch.allclose(depth.sum(2), torch.ones(2, 3, 4, 5))

import torch
from lss.models.lss import LSS


def test_lss_returns_bev_and_depth():
    model = LSS(depth_bins=2, bev_bounds=((-2., 2., 1.), (-2., 2., 1.), (-1., 1., 1.)))
    images = torch.randn(1, 1, 3, 8, 8)
    bev, depth = model(images, torch.eye(3)[None,None], torch.eye(4)[None,None], [1., 2.])
    assert bev.shape == (1, 64, 4, 4)
    assert depth.shape == (1, 1, 2, 2, 2)

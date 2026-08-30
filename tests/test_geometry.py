import torch
from lss.geometry import make_frustum


def test_frustum_has_depth_and_pixel_coordinates():
    out = make_frustum(2, 3, [1.0, 2.0])
    assert out.shape == (2, 2, 3, 3)
    assert torch.equal(out[0, 0, 0], torch.tensor([0., 0., 1.]))

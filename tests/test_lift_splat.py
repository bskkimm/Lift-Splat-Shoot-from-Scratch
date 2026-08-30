import torch
from lss.models.lift_splat import LiftSplat


def test_lift_splat_accumulates_features_in_bev():
    depth = torch.ones(1, 1, 1, 1, 2)
    context = torch.tensor([[[[[2., 3.]]]]])
    points = torch.tensor([[[[[[0., 0., 0.], [1., 0., 0.]]]]]])
    out = LiftSplat((0., 2., 1.), (0., 1., 1.), (0., 1., 1.))(depth, context, points)
    assert torch.equal(out[0, 0, 0], torch.tensor([2., 3.]))

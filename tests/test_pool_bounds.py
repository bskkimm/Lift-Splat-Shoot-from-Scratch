import torch
from lss.models.lift_splat import LiftSplat
def test_pool_ignores_points_outside_z_bounds():
    layer = LiftSplat((0,2,1),(0,1,1),(0,1,1)); d=torch.ones(1,1,1,1,1); c=torch.ones(1,1,1,1,1); p=torch.tensor([[[[[[0.,0.,2.]]]]]])
    assert layer(d,c,p).sum() == 0

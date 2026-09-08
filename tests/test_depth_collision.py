import torch
from lss.depth_targets import make_depth_target
def test_depth_target_keeps_nearest_collision():
    points=torch.tensor([[[[1.,1.,4.],[1.,1.,2.]]]]); assert make_depth_target(points,torch.zeros(1,1,3,3))[0,0,1,1] == 2

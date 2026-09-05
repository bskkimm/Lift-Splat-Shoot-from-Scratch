import torch
from lss.depth_targets import make_depth_target
def test_depth_target_projects_points():
    out=make_depth_target(torch.tensor([[[[1.,2.,3.]]]]),torch.zeros(1,1,4,4)); assert out[0,0,2,1] == 3

import torch
from lss.losses import depth_loss, detection_loss
def test_losses_are_finite():
    assert depth_loss(torch.randn(2,4,3,3), torch.zeros(2,3,3,dtype=torch.long)).isfinite()
    assert detection_loss(torch.zeros(2,9,3,3), torch.ones(2,9,3,3)).item() > 0

import torch
from lss.losses import detection_loss
def test_detection_loss_is_zero_for_empty_targets(): assert detection_loss(torch.ones(1),torch.empty(0)).item() == 0

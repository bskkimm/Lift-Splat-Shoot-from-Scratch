import torch
from lss.losses import focal_loss
def test_focal_loss_backpropagates():
    logits = torch.zeros(2, requires_grad=True); loss = focal_loss(logits, torch.tensor([1., 0.])); loss.backward(); assert logits.grad is not None and torch.isfinite(loss)

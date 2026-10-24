import torch
from lss.optim import build_optimizer
def test_optimizer_builder_returns_adamw(): assert isinstance(build_optimizer(torch.nn.Linear(1,1)),torch.optim.AdamW)

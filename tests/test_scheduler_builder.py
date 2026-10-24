import torch
from lss.optim import build_optimizer, build_scheduler
def test_scheduler_builder_returns_cosine_schedule():
    m=torch.nn.Linear(1,1); assert isinstance(build_scheduler(build_optimizer(m),2),torch.optim.lr_scheduler.CosineAnnealingLR)

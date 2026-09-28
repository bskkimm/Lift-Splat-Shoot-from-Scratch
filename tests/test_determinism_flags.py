import torch
from lss.reproducibility import seed_everything
def test_seed_sets_deterministic_flags():
    seed_everything(1); assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark

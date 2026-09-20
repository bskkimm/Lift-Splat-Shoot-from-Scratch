import torch
from lss.reproducibility import seed_everything
def test_seed_everything_repeats_random_tensor():
    seed_everything(3); first=torch.rand(2); seed_everything(3); assert torch.equal(first,torch.rand(2))

import pytest, torch
from lss.optim import build_optimizer
def test_optimizer_rejects_invalid_learning_rate():
    with pytest.raises(ValueError): build_optimizer(torch.nn.Linear(1,1),lr=0)

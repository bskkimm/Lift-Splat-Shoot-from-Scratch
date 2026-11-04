import pytest, torch
from lss.optim import build_optimizer
def test_optimizer_rejects_frozen_model():
    model=torch.nn.Linear(1,1)
    for p in model.parameters(): p.requires_grad=False
    with pytest.raises(ValueError): build_optimizer(model)

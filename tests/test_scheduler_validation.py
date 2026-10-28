import pytest, torch
from lss.optim import build_scheduler
def test_scheduler_rejects_invalid_epochs():
    with pytest.raises(ValueError): build_scheduler(torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],.1),0)

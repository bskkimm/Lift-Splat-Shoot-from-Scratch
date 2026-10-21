import torch
from lss.models.diagnostics import trainable_parameter_count
def test_trainable_parameter_count(): assert trainable_parameter_count(torch.nn.Linear(2,1)) == 3

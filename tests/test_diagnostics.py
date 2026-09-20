import torch
from lss.models.diagnostics import parameter_count
def test_parameter_count_respects_trainable_flag():
    model=torch.nn.Linear(2,1); model.bias.requires_grad=False; assert parameter_count(model)==2 and parameter_count(model,False)==3

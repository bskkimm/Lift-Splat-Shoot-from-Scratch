import torch
from lss.engine import train_step
def test_train_step_updates_parameters():
    model = torch.nn.Linear(2,1); before = model.weight.detach().clone(); opt = torch.optim.SGD(model.parameters(), .1)
    train_step(model, opt, (torch.ones(1,2),), lambda out: (out**2).mean()); assert not torch.equal(before, model.weight)

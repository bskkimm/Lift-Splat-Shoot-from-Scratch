import torch
from lss.engine import fit
def test_fit_returns_epoch_history():
    model = torch.nn.Linear(1,1); opt = torch.optim.SGD(model.parameters(), .1); loader = [((torch.ones(1,1),), torch.ones(1,1))]
    assert len(fit(model, loader, opt, lambda out, target: ((out-target)**2).mean(), 2)) == 2

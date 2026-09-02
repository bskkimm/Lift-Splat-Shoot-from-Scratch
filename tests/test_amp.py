import torch
from lss.engine import fit
def test_fit_supports_amp_flag_on_cpu():
    model=torch.nn.Linear(1,1); opt=torch.optim.SGD(model.parameters(),.1); loader=[((torch.ones(1,1),),torch.ones(1,1))]
    assert len(fit(model,loader,opt,lambda o,t:((o-t)**2).mean(),1,use_amp=False)) == 1

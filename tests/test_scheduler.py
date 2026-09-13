import torch
from lss.engine import fit
def test_fit_steps_scheduler():
    model=torch.nn.Linear(1,1); opt=torch.optim.SGD(model.parameters(),.1); scheduler=torch.optim.lr_scheduler.StepLR(opt,1,.5); loader=[((torch.ones(1,1),),torch.ones(1,1))]; fit(model,loader,opt,lambda o,t:((o-t)**2).mean(),1,scheduler=scheduler); assert opt.param_groups[0]["lr"] == .05

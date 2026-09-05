import torch
from lss.checkpoint import resume_checkpoint
def test_resume_returns_saved_epoch(tmp_path):
    model=torch.nn.Linear(1,1); opt=torch.optim.SGD(model.parameters(),.1); path=tmp_path/"x.pt"; torch.save({"epoch":4,"state_dict":model.state_dict(),"optimizer":opt.state_dict()},path); assert resume_checkpoint(model,opt,path)==4

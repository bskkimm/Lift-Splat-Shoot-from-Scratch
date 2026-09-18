import torch
from lss.checkpoint import resume_checkpoint
def test_resume_restores_scheduler(tmp_path):
    m=torch.nn.Linear(1,1); o=torch.optim.SGD(m.parameters(),.1); s=torch.optim.lr_scheduler.StepLR(o,1); p=tmp_path/"c.pt"; torch.save({"epoch":2,"state_dict":m.state_dict(),"optimizer":o.state_dict(),"scheduler":s.state_dict()},p); s2=torch.optim.lr_scheduler.StepLR(o,1); assert resume_checkpoint(m,o,p,scheduler=s2)==2

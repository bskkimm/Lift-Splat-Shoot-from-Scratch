import torch
from lss.engine import save_checkpoint
def test_checkpoint_can_store_scheduler_state(tmp_path):
    model=torch.nn.Linear(1,1); opt=torch.optim.SGD(model.parameters(),.1); sch=torch.optim.lr_scheduler.StepLR(opt,1); path=tmp_path/"c.pt"; save_checkpoint(model,opt,path,1,sch); assert "scheduler" in torch.load(path)

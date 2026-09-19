import torch
from lss.engine import save_checkpoint
def test_save_checkpoint_stores_metadata(tmp_path):
    p=tmp_path/"c.pt"; m=torch.nn.Linear(1,1); o=torch.optim.SGD(m.parameters(),.1); save_checkpoint(m,o,p,1,metadata={"seed":7}); assert torch.load(p)["metadata"]["seed"]==7

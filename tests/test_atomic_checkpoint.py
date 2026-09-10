import torch
from lss.engine import save_checkpoint
def test_save_checkpoint_leaves_no_temporary_file(tmp_path):
    path=tmp_path/"checkpoint.pt"; save_checkpoint(torch.nn.Linear(1,1),torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],.1),path,1); assert path.exists() and not (tmp_path/"checkpoint.pt.tmp").exists()

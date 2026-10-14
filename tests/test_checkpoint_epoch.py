import torch
from lss.checkpoint import checkpoint_epoch
def test_checkpoint_epoch_is_integer(tmp_path):
    p=tmp_path/"c.pt"; torch.save({"epoch":"3"},p); assert checkpoint_epoch(p)==3

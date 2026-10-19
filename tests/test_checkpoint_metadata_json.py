import torch
from lss.checkpoint import checkpoint_metadata_json
def test_checkpoint_metadata_is_json(tmp_path):
    p=tmp_path/"c.pt"; torch.save({"epoch":2},p); assert '"epoch": 2' in checkpoint_metadata_json(p)

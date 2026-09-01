import torch
from lss.checkpoint import checkpoint_metadata
def test_checkpoint_metadata_excludes_tensors(tmp_path):
    path = tmp_path / "x.pt"; torch.save({"epoch": 3, "state_dict": {}, "tensor": torch.ones(1)}, path); assert checkpoint_metadata(path) == {"epoch": 3}

import torch
from lss.checkpoint import load_checkpoint
def test_checkpoint_loader_accepts_state_dict_wrapper(tmp_path):
    model = torch.nn.Linear(2, 1); path = tmp_path / "model.pt"
    torch.save({"state_dict": {"module.weight": model.weight, "module.bias": model.bias}}, path)
    target = torch.nn.Linear(2, 1); load_checkpoint(target, path); assert torch.equal(target.weight, model.weight)

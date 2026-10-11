import torch
from pathlib import Path


def export_state_dict(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)

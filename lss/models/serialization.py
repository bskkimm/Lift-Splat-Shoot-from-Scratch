import torch
from pathlib import Path


def export_state_dict(model, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = str(path) + ".tmp"
    torch.save({"state_dict": model.state_dict()}, temporary); Path(temporary).replace(path)

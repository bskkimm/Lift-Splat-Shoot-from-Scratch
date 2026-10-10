import torch


def export_state_dict(model, path):
    torch.save({"state_dict": model.state_dict()}, path)

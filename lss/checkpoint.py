import torch


def load_checkpoint(model, path, strict=True, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    state = state.get("state_dict", state)
    translated = {key.removeprefix("module."): value for key, value in state.items()}
    if not translated:
        raise ValueError("checkpoint contains no model parameters")
    return model.load_state_dict(translated, strict=strict)


def checkpoint_metadata(path, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    return {key: value for key, value in state.items() if key != "state_dict" and not torch.is_tensor(value)}


def resume_checkpoint(model, optimizer, path, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    load_checkpoint(model, path, map_location=map_location)
    if "optimizer" in state: optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0))

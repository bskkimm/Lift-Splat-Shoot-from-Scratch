"""Atomic epoch-boundary checkpoints with explicit random-generator state.

This module saves and restores state only. Exact continuation additionally
requires compatible model/data/configuration and a caller-controlled loader
whose sampler uses the supplied generator. Mid-epoch and worker/prefetch state
are not captured.
"""

import copy
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def _config_json(config):
    if not isinstance(config, dict):
        raise TypeError("checkpoint config must be a dictionary")
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _type_name(value):
    if value is None:
        return None
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _snapshot_on_cpu(value):
    """Clone checkpoint state onto CPU without creating a second GPU copy."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        result = copy.copy(value)
        for key, item in value.items():
            result[key] = _snapshot_on_cpu(item)
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, list):
        return [_snapshot_on_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_on_cpu(item) for item in value)
    return copy.deepcopy(value)


def _preflight_structure(model, optimizer, state):
    """Reject common incompatible structures before loading any weights."""
    current = model.state_dict()
    saved = state["model"]
    if current.keys() != saved.keys():
        raise ValueError("checkpoint model state keys do not match")
    for key, value in current.items():
        other = saved[key]
        if isinstance(value, torch.Tensor):
            if not isinstance(other, torch.Tensor) or value.shape != other.shape or value.dtype != other.dtype:
                raise ValueError(f"checkpoint model tensor shape or dtype mismatch: {key}")
        elif isinstance(other, torch.Tensor):
            raise ValueError(f"checkpoint model state type mismatch: {key}")
    current_groups = optimizer.state_dict()["param_groups"]
    saved_groups = state["optimizer"]["param_groups"]
    if len(current_groups) != len(saved_groups) or any(
        len(current_group["params"]) != len(saved_group["params"])
        for current_group, saved_group in zip(current_groups, saved_groups)
    ):
        raise ValueError("checkpoint optimizer parameter groups do not match")


def save_epoch_checkpoint(path, model, optimizer, epoch, config, loader_generator,
                          scheduler=None, capture_cuda=False):
    """Atomically save model, optimizer and RNG state at an epoch boundary."""
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if not isinstance(loader_generator, torch.Generator):
        raise TypeError("loader_generator must be a torch.Generator")
    config_json = _config_json(config)
    if capture_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA RNG capture requested but CUDA is unavailable")
    numpy_state = np.random.get_state()
    state = {
        "format_version": 1,
        "epoch": epoch,
        "config_json": config_json,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "types": {"model": _type_name(model), "optimizer": _type_name(optimizer),
                  "scheduler": _type_name(scheduler)},
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                      int(numpy_state[3]), float(numpy_state[4])),
        "torch_rng": torch.get_rng_state(),
        "loader_rng": loader_generator.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if capture_cuda else None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_epoch_checkpoint(path, model, optimizer, config, loader_generator,
                          scheduler=None, map_location="cpu"):
    """Restore a compatible checkpoint; return the completed epoch number."""
    if not isinstance(loader_generator, torch.Generator):
        raise TypeError("loader_generator must be a torch.Generator")
    state = torch.load(path, map_location=map_location, weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if state["config_json"] != _config_json(config):
        raise ValueError("checkpoint config does not match the requested config")
    if (state["scheduler"] is None) != (scheduler is None):
        raise ValueError("checkpoint scheduler presence does not match")
    for name, component in (("model", model), ("optimizer", optimizer), ("scheduler", scheduler)):
        if state["types"][name] != _type_name(component):
            raise ValueError(f"checkpoint {name} type does not match")
    cuda_rng = state["cuda_rng"]
    if cuda_rng is not None and (not torch.cuda.is_available() or len(cuda_rng) != torch.cuda.device_count()):
        raise RuntimeError("checkpoint CUDA RNG state is incompatible with available CUDA devices")
    _preflight_structure(model, optimizer, state)
    previous_model = _snapshot_on_cpu(model.state_dict())
    previous_optimizer = _snapshot_on_cpu(optimizer.state_dict())
    previous_scheduler = _snapshot_on_cpu(scheduler.state_dict()) if scheduler is not None else None
    previous_python = random.getstate()
    previous_numpy = np.random.get_state()
    previous_torch = torch.get_rng_state()
    previous_loader = loader_generator.get_state()
    previous_cuda = torch.cuda.get_rng_state_all() if cuda_rng is not None else None
    try:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        numpy_state = state["numpy_rng"]
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                             numpy_state[2], numpy_state[3], numpy_state[4]))
        torch.set_rng_state(state["torch_rng"].cpu())
        loader_generator.set_state(state["loader_rng"].cpu())
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_rng])
    except BaseException:
        model.load_state_dict(previous_model)
        optimizer.load_state_dict(previous_optimizer)
        if scheduler is not None:
            scheduler.load_state_dict(previous_scheduler)
        random.setstate(previous_python)
        np.random.set_state(previous_numpy)
        torch.set_rng_state(previous_torch)
        loader_generator.set_state(previous_loader)
        if previous_cuda is not None:
            torch.cuda.set_rng_state_all(previous_cuda)
        raise
    return state["epoch"]

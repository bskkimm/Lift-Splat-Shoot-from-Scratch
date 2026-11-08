"""LSS-style scoring and classification of BEV trajectory templates."""

import torch
from torch.nn import functional as F


def trajectory_logits(cost_maps, cells):
    """Return negative summed costs for indexed templates.

    ``cost_maps`` is floating [B,X,Y]. ``cells`` is integer [B,K,T,2], with
    coordinates ordered (x,y). Every point must lie in bounds. No masking,
    metre-to-grid conversion, interpolation, or time weighting is applied.
    A softmax over K implements the paper's trajectory distribution.
    """
    if cost_maps.ndim != 3 or cells.ndim != 4 or cells.shape[-1] != 2:
        raise ValueError("expected cost maps [B,X,Y] and cells [B,K,T,2]")
    if cost_maps.shape[0] != cells.shape[0] or cells.shape[0] == 0 or cells.shape[1] == 0 or cells.shape[2] == 0:
        raise ValueError("trajectory batches and dimensions must be nonempty and aligned")
    if not cost_maps.is_floating_point():
        raise TypeError("cost maps must be floating point")
    if cells.dtype not in (torch.int32, torch.int64):
        raise TypeError("trajectory cells must use integer indices")
    if cells.device != cost_maps.device:
        raise ValueError("trajectory cells and cost maps must share a device")
    x, y = cells[..., 0], cells[..., 1]
    if ((x < 0) | (x >= cost_maps.shape[1]) | (y < 0) | (y >= cost_maps.shape[2])).any():
        raise ValueError("trajectory cell is outside the cost map")
    batch = torch.arange(cost_maps.shape[0], device=cost_maps.device)[:, None, None]
    return -cost_maps[batch, x.long(), y.long()].sum(dim=-1)


def nearest_template(templates, expert):
    """Return nearest template index under summed squared XY distance.

    Templates [B,K,T,2] and expert trajectories [B,T,2] must use the same
    continuous coordinate frame and units; each timestep is weighted equally.
    """
    if templates.ndim != 4 or templates.shape[-1] != 2 or expert.shape != (templates.shape[0], templates.shape[2], 2):
        raise ValueError("expected templates [B,K,T,2] and expert [B,T,2]")
    if templates.shape[0] == 0 or templates.shape[1] == 0 or templates.shape[2] == 0:
        raise ValueError("trajectory dimensions must be nonempty")
    if not templates.is_floating_point() or not expert.is_floating_point():
        raise TypeError("trajectory coordinates must be floating point")
    if templates.device != expert.device:
        raise ValueError("templates and expert must share a device")
    if not torch.isfinite(templates).all() or not torch.isfinite(expert).all():
        raise ValueError("trajectory coordinates must be finite")
    return (templates - expert[:, None]).square().sum(dim=(-1, -2)).argmin(dim=1)


def trajectory_classification_loss(logits, templates, expert):
    """Cross-entropy against the nearest template, as in LSS section 3.3."""
    if logits.ndim != 2 or logits.shape != templates.shape[:2]:
        raise ValueError("logits must have shape [B,K] matching templates")
    return F.cross_entropy(logits, nearest_template(templates, expert))

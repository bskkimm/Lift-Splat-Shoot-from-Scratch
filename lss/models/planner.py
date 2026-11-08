"""LSS-style BEV cost-map and trajectory-template scoring composition."""

import math

import torch
from torch import nn

from .segmentation import LSSSegmentation
from .trajectory import trajectory_logits


def metric_trajectory_cells(trajectories, x_bounds, y_bounds):
    """Map ego-frame XY metres [B,K,T,2] to half-open BEV grid cells.

    Coordinates on or outside an upper bound are invalid. Inputs must already
    be expressed in the same ego frame as the camera geometry and BEV grid.
    """
    if trajectories.ndim != 4 or trajectories.shape[-1] != 2 or any(size == 0 for size in trajectories.shape[:-1]):
        raise ValueError("trajectories must have nonempty shape [B,K,T,2]")
    if not trajectories.is_floating_point() or not torch.isfinite(trajectories).all():
        raise ValueError("trajectory coordinates must be finite floating point values")
    sizes = []
    for lower, upper, step in (x_bounds, y_bounds):
        if not all(math.isfinite(value) for value in (lower, upper, step)) or step <= 0 or upper <= lower:
            raise ValueError("grid bounds must be finite and increasing")
        count = (upper - lower) / step
        if round(count) < 1 or not math.isclose(count, round(count), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("grid bounds must define an integral number of bins")
        sizes.append(int(round(count)))
    lower = trajectories.new_tensor([x_bounds[0], y_bounds[0]])
    upper = trajectories.new_tensor([x_bounds[1], y_bounds[1]])
    step = trajectories.new_tensor([x_bounds[2], y_bounds[2]])
    if not ((trajectories >= lower) & (trajectories < upper)).all():
        raise ValueError("trajectory coordinates are outside the BEV grid")
    cells = torch.floor((trajectories - lower) / step).long()
    max_index = torch.tensor([sizes[0] - 1, sizes[1] - 1], device=trajectories.device)
    return torch.minimum(cells.clamp_min(0), max_index)


class LSSPlanner(nn.Module):
    """Predict a BEV cost map and a probability for each trajectory template.

    This is a model primitive; template fitting, data association, and planning
    evaluation are not included. Trajectories are ego-frame metric XY points.
    """

    def __init__(self, x_bounds=(-50., 50., 0.5), y_bounds=(-50., 50., 0.5), **model_kwargs):
        super().__init__()
        self.cost_model = LSSSegmentation(x_bounds=x_bounds, y_bounds=y_bounds, out_channels=1, **model_kwargs)

    def forward(self, images, intrinsics, extrinsics, trajectories, post_rots=None, post_trans=None):
        cost = self.cost_model(images, intrinsics, extrinsics, post_rots=post_rots, post_trans=post_trans)[:, 0]
        bounds = self.cost_model.voxel_pool.bounds
        cells = metric_trajectory_cells(trajectories, bounds[0], bounds[1])
        logits = trajectory_logits(cost, cells)
        return {"cost_map": cost, "logits": logits, "probabilities": logits.softmax(dim=1)}

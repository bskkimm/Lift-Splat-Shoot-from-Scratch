"""Differentiable multi-camera voxel summation for LSS-style BEV features."""

import math

import torch
from torch import nn


class VoxelPool(nn.Module):
    """Pool ego-frame points into [B, Z*C, X, Y] with z-major channels.

    Points have shape [B,N,D,H,W,3], in metres, and lifted features have
    shape [B,N,D,H,W,C]. Each grid interval is half-open: [lower, upper).
    Gradients flow through lifted features, not discrete voxel indices.
    """

    def __init__(self, x_bounds, y_bounds, z_bounds):
        super().__init__()
        bounds = (x_bounds, y_bounds, z_bounds)
        sizes = []
        for lower, upper, step in bounds:
            if not all(math.isfinite(value) for value in (lower, upper, step)) or step <= 0 or upper <= lower:
                raise ValueError("voxel bounds must be finite and increasing with positive step")
            count = (upper - lower) / step
            if round(count) < 1 or not math.isclose(count, round(count), rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError("voxel bounds must define an integral number of bins")
            sizes.append(int(round(count)))
        self.grid_size = tuple(sizes)
        # Keep the configured values independent of module device/dtype casts.
        self.bounds = tuple(tuple(float(value) for value in bound) for bound in bounds)

    def get_extra_state(self):
        return {"bounds": self.bounds}

    def set_extra_state(self, state):
        if state.get("bounds") != self.bounds:
            raise RuntimeError("checkpoint voxel bounds do not match the configured bounds")

    def forward(self, points, features):
        if points.ndim != 6 or points.shape[-1] != 3 or features.ndim != 6:
            raise ValueError("points and lifted features must have six dimensions")
        if not points.is_floating_point() or not features.is_floating_point():
            raise TypeError("points and lifted features must use floating point dtype")
        if points.shape[:-1] != features.shape[:-1] or points.device != features.device:
            raise ValueError("points and lifted features must share sample dimensions and device")
        batch = points.shape[0]
        channels = features.shape[-1]
        nx, ny, nz = self.grid_size
        lower = points.new_tensor([bound[0] for bound in self.bounds])
        upper = points.new_tensor([bound[1] for bound in self.bounds])
        step = points.new_tensor([bound[2] for bound in self.bounds])
        valid = ((points >= lower) & (points < upper)).all(dim=-1)
        selected_points = points[valid]
        xyz = torch.floor((selected_points - lower) / step).long()
        # Floating-point division can round an in-bounds point to the bin count.
        max_index = torch.tensor((nx - 1, ny - 1, nz - 1), device=points.device)
        xyz = torch.minimum(xyz.clamp_min(0), max_index)
        batch_index = torch.arange(batch, device=points.device).view(batch, 1, 1, 1, 1)
        batch_index = batch_index.expand(points.shape[:-1])[valid]
        index = ((batch_index * nz + xyz[:, 2]) * nx + xyz[:, 0]) * ny + xyz[:, 1]
        flat = features.new_zeros((batch * nz * nx * ny, channels))
        flat = flat.index_add(0, index, features[valid])
        # Reference ordering: concatenate each z plane's C channels in sequence.
        return flat.reshape(batch, nz, nx, ny, channels).permute(0, 1, 4, 2, 3).reshape(batch, nz * channels, nx, ny)

import torch
import torch.nn as nn


class LiftSplat(nn.Module):
    def __init__(self, x_bounds=(-50., 50., 1.), y_bounds=(-50., 50., 1.), z_bounds=(-5., 3., 1.)):
        super().__init__()
        self.register_buffer("bounds", torch.tensor([x_bounds, y_bounds, z_bounds]))

    def forward(self, depth, context, points):
        # depth/context: [B,C,D,H,W], [B,C,F,H,W]; points: [B,C,D,H,W,3]
        b, c, d, h, w = depth.shape
        out = depth.unsqueeze(2) * context.unsqueeze(3)
        xyz = points.round().long()
        x = ((xyz[..., 0] - self.bounds[0, 0]) / self.bounds[0, 2]).long()
        y = ((xyz[..., 1] - self.bounds[1, 0]) / self.bounds[1, 2]).long()
        nx = int((self.bounds[0, 1] - self.bounds[0, 0]) / self.bounds[0, 2])
        ny = int((self.bounds[1, 1] - self.bounds[1, 0]) / self.bounds[1, 2])
        result = depth.new_zeros(b, out.shape[2], ny * nx)
        valid = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        valid = valid & (xyz[..., 2] >= self.bounds[2, 0]) & (xyz[..., 2] < self.bounds[2, 1])
        linear = (y * nx + x).reshape(b, c, 1, -1).expand(b, c, out.shape[2], -1)
        weights = out.reshape(b, c, out.shape[2], -1)
        valid = valid.reshape(b, c, 1, -1).expand_as(weights)
        for fi in range(out.shape[2]):
            idx = linear[:, :, fi].reshape(b, -1).masked_fill(~valid[:, :, fi].reshape(b, -1), 0)
            result[:, fi].scatter_add_(1, idx, weights[:, :, fi].reshape(b, -1) * valid[:, :, fi].reshape(b, -1))
        return result.reshape(b, out.shape[2], ny, nx)

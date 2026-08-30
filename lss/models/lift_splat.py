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
        result = depth.new_zeros(b, out.shape[2], ny, nx)
        valid = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        for bi in range(b):
            for ci in range(c):
                for di in range(d):
                    mask = valid[bi, ci, di]
                    for fi in range(out.shape[2]):
                        result[bi, fi].index_put_((y[bi, ci, di][mask], x[bi, ci, di][mask]), out[bi, ci, fi, di][mask], accumulate=True)
        return result

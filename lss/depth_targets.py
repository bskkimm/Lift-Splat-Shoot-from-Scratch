import torch

def make_depth_target(points, depths):
    target = torch.zeros_like(depths)
    pixels = points[..., :2].long()
    valid = (pixels[..., 0] >= 0) & (pixels[..., 1] >= 0) & (pixels[..., 0] < depths.shape[-1]) & (pixels[..., 1] < depths.shape[-2])
    target[..., pixels[..., 1][valid], pixels[..., 0][valid]] = points[..., 2][valid]
    return target

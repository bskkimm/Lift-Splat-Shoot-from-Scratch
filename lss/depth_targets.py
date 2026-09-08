import torch

def make_depth_target(points, depths):
    target = torch.zeros_like(depths)
    pixels = points[..., :2].long()
    valid = (pixels[..., 0] >= 0) & (pixels[..., 1] >= 0) & (pixels[..., 0] < depths.shape[-1]) & (pixels[..., 1] < depths.shape[-2])
    for x, y, z in zip(pixels[..., 0][valid], pixels[..., 1][valid], points[..., 2][valid]):
        if target[..., y, x].item() == 0 or z < target[..., y, x]: target[..., y, x] = z
    return target

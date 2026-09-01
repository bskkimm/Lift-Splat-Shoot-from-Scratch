import torch


def make_frustum(height, width, depths, device=None):
    d = torch.as_tensor(depths, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device))
    return torch.stack((x[None].expand(len(d), -1, -1), y[None].expand(len(d), -1, -1), d[:, None, None].expand(-1, height, width)), -1)


def camera_to_ego(points, intrinsics, extrinsics):
    if points.ndim != 6 or intrinsics.shape[:2] != points.shape[:2] or extrinsics.shape[:2] != points.shape[:2]:
        raise ValueError("points, intrinsics, and extrinsics must share batch and camera axes")
    b, c, d, h, w, _ = points.shape
    pix = points.reshape(b, c, -1, 3)
    xyz = torch.linalg.solve(intrinsics, torch.cat((pix[..., :2] * pix[..., 2:3], pix[..., 2:3]), -1).transpose(-1, -2)).transpose(-1, -2)
    return ((extrinsics[..., :3, :3].unsqueeze(2) @ xyz.unsqueeze(-1)).squeeze(-1) + extrinsics[..., :3, 3].unsqueeze(2)).reshape(b, c, d, h, w, 3)

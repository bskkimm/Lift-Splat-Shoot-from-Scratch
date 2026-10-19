import torch

__all__ = ["make_frustum", "camera_to_ego", "quaternion_to_matrix"]


def quaternion_to_matrix(q):
    q = torch.as_tensor(q); dtype = q.dtype if q.is_floating_point() else torch.float32; q = q.to(dtype); w, x, y, z = q
    return torch.stack((torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w))), torch.stack((2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w))), torch.stack((2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))))


def make_frustum(height, width, depths, device=None):
    if height <= 0 or width <= 0: raise ValueError("frustum dimensions must be positive")
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

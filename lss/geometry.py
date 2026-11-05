import torch

__all__ = ["make_frustum", "camera_to_ego", "quaternion_to_matrix"]


def quaternion_to_matrix(q):
    q = torch.as_tensor(q); dtype = q.dtype if q.is_floating_point() else torch.float32; q = q.to(dtype); w, x, y, z = q
    return torch.stack((torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w))), torch.stack((2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w))), torch.stack((2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))))


def make_frustum(height, width, depths, device=None, image_size=None):
    """Return [D,H,W,3] frustum points in input-image pixel coordinates.

    ``height`` and ``width`` are feature-map dimensions; ``image_size`` is
    the input image (height, width) corresponding to the camera intrinsics.
    When omitted, feature-map coordinates are used for compatibility.
    """
    if height <= 0 or width <= 0: raise ValueError("frustum dimensions must be positive")
    image_height, image_width = image_size if image_size is not None else (height, width)
    if image_height <= 0 or image_width <= 0: raise ValueError("image dimensions must be positive")
    d = torch.as_tensor(depths, device=device)
    if not d.is_floating_point():
        d = d.to(dtype=torch.float32)
    y, x = torch.meshgrid(
        torch.linspace(0, image_height - 1, height, dtype=d.dtype, device=d.device),
        torch.linspace(0, image_width - 1, width, dtype=d.dtype, device=d.device),
    )
    return torch.stack((x[None].expand(len(d), -1, -1), y[None].expand(len(d), -1, -1), d[:, None, None].expand(-1, height, width)), -1)


def camera_to_ego(points, intrinsics, extrinsics, post_rots=None, post_trans=None):
    """Unproject frustum points; optionally undo paired image post-transforms.

    Post-transforms map pre-augmentation pixel/depth coordinates to the
    coordinates in ``points``. Intrinsics must describe pre-augmentation pixels
    when the post-transform pair is supplied.
    """
    if points.ndim != 6 or intrinsics.shape[:2] != points.shape[:2] or extrinsics.shape[:2] != points.shape[:2]:
        raise ValueError("points, intrinsics, and extrinsics must share batch and camera axes")
    b, c, d, h, w, _ = points.shape
    pix = points.reshape(b, c, -1, 3)
    if (post_rots is None) != (post_trans is None):
        raise ValueError("post_rots and post_trans must be provided together")
    if post_rots is not None:
        if post_rots.shape != (b, c, 3, 3) or post_trans.shape != (b, c, 3):
            raise ValueError("post-transform shape must be [B,N,3,3] and [B,N,3]")
        pix = torch.linalg.solve(post_rots, (pix - post_trans.unsqueeze(2)).transpose(-1, -2)).transpose(-1, -2)
    xyz = torch.linalg.solve(intrinsics, torch.cat((pix[..., :2] * pix[..., 2:3], pix[..., 2:3]), -1).transpose(-1, -2)).transpose(-1, -2)
    return ((extrinsics[..., :3, :3].unsqueeze(2) @ xyz.unsqueeze(-1)).squeeze(-1) + extrinsics[..., :3, 3].unsqueeze(2)).reshape(b, c, d, h, w, 3)

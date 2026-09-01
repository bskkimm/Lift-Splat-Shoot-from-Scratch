import pytest
import torch
from lss.geometry import camera_to_ego


def test_camera_to_ego_supports_batched_cameras():
    points = torch.zeros(2, 3, 1, 1, 1, 3); points[..., 2] = 2
    out = camera_to_ego(points, torch.eye(3).expand(2,3,3,3), torch.eye(4).expand(2,3,4,4))
    assert out.shape == points.shape and torch.equal(out[..., 2], points[..., 2])


def test_camera_to_ego_rejects_mismatched_axes():
    with pytest.raises(ValueError): camera_to_ego(torch.zeros(1,1,1,1,1,3), torch.eye(3)[None,None], torch.eye(4)[None,None].expand(2,1,4,4))

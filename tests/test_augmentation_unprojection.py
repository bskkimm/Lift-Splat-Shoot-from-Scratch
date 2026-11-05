import pytest
import torch

from lss.geometry import camera_to_ego
from lss.models.lss import LSS
from lss.models.detector import LSSDetector


def test_unprojection_undoes_post_augmentation_before_intrinsics():
    points = torch.tensor([[[[[[14.0, 11.0, 2.0]]]]]])
    post_rots = torch.diag(torch.tensor([2.0, 2.0, 1.0]))[None, None]
    post_trans = torch.tensor([[[10.0, 5.0, 0.0]]])

    actual = camera_to_ego(points, torch.eye(3)[None, None], torch.eye(4)[None, None],
                           post_rots=post_rots, post_trans=post_trans)

    torch.testing.assert_allclose(actual[0, 0, 0, 0, 0], torch.tensor([4.0, 6.0, 2.0]))


def test_model_passes_post_augmentation_into_unprojection():
    model = LSS(depth_bins=1)
    captured = []
    handle = model.lift_splat.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[2].detach()))
    try:
        model(torch.randn(1, 1, 3, 8, 12), torch.eye(3)[None, None], torch.eye(4)[None, None],
              [1.0], post_rots=torch.eye(3)[None, None], post_trans=torch.tensor([[[1.0, 2.0, 0.0]]]))
    finally:
        handle.remove()

    torch.testing.assert_allclose(captured[0][0, 0, 0, -1, -1], torch.tensor([10.0, 5.0, 1.0]))


def test_unprojection_requires_matching_augmentation_pair():
    points = torch.zeros(1, 1, 1, 1, 1, 3)
    with pytest.raises(ValueError, match="together"):
        camera_to_ego(points, torch.eye(3)[None, None], torch.eye(4)[None, None],
                      post_rots=torch.eye(3)[None, None])


def test_unprojection_rejects_malformed_augmentation_shapes():
    points = torch.zeros(1, 1, 1, 1, 1, 3)
    with pytest.raises(ValueError, match="shape"):
        camera_to_ego(points, torch.eye(3)[None, None], torch.eye(4)[None, None],
                      post_rots=torch.eye(3)[None, None], post_trans=torch.zeros(1, 3))


def test_detector_forwards_post_augmentation_to_pooling():
    model = LSSDetector(depth_bins=1)
    captured = []
    handle = model.lss.lift_splat.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[2].detach()))
    try:
        model(torch.randn(1, 1, 3, 8, 12), torch.eye(3)[None, None], torch.eye(4)[None, None],
              [1.0], post_rots=torch.eye(3)[None, None], post_trans=torch.tensor([[[1.0, 2.0, 0.0]]]))
    finally:
        handle.remove()
    torch.testing.assert_allclose(captured[0][0, 0, 0, -1, -1], torch.tensor([10.0, 5.0, 1.0]))

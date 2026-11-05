import torch

from lss.geometry import camera_to_ego, make_frustum
from lss.models.lss import LSS


def test_frustum_uses_image_pixel_coordinates_for_low_resolution_features():
    frustum = make_frustum(2, 3, [2.0], image_size=(8, 12))

    torch.testing.assert_allclose(frustum[0, 0, :, 0], torch.tensor([0.0, 5.5, 11.0]))
    torch.testing.assert_allclose(frustum[0, :, 0, 1], torch.tensor([0.0, 7.0]))

    intrinsics = torch.diag(torch.tensor([2.0, 2.0, 1.0]))[None, None]
    points = camera_to_ego(frustum[None, None], intrinsics, torch.eye(4)[None, None])
    torch.testing.assert_allclose(points[0, 0, 0, 1, 2], torch.tensor([11.0, 7.0, 2.0]))


def test_frustum_preserves_floating_depth_dtype():
    depth = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert make_frustum(2, 2, depth).dtype == torch.float64


def test_lss_passes_original_image_coordinates_to_pooling():
    model = LSS(depth_bins=1)
    captured = []
    handle = model.lift_splat.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[2].detach()))
    try:
        model(torch.randn(1, 1, 3, 8, 12), torch.eye(3)[None, None],
              torch.eye(4)[None, None], [1.0])
    finally:
        handle.remove()

    torch.testing.assert_allclose(captured[0][0, 0, 0, -1, -1], torch.tensor([11.0, 7.0, 1.0]))

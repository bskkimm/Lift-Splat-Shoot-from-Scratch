import pytest
import torch


def make_model():
    from lss.models.segmentation import LSSSegmentation

    return LSSSegmentation(
        x_bounds=(-8, 8, 1), y_bounds=(-8, 8, 1), z_bounds=(0, 3, 3),
        depth_bounds=(1, 3, 1), context_channels=8, pretrained=False,
    )


def synthetic_inputs():
    images = torch.randn(1, 2, 3, 32, 64)
    k = torch.tensor([[10., 0., 32.], [0., 10., 16.], [0., 0., 1.]])
    intrinsics = k[None, None].expand(1, 2, 3, 3).clone()
    extrinsics = torch.eye(4)[None, None].expand(1, 2, 4, 4).clone()
    return images, intrinsics, extrinsics


def test_segmentation_forward_and_bce_backpropagate_through_model():
    from lss.losses import binary_segmentation_loss

    torch.manual_seed(0)
    model = make_model()
    images, intrinsics, extrinsics = synthetic_inputs()
    images.requires_grad_()

    logits = model(images, intrinsics, extrinsics)
    loss = binary_segmentation_loss(logits, torch.ones_like(logits))
    loss.backward()

    assert logits.shape == (1, 1, 16, 16)
    assert torch.isfinite(loss)
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert torch.count_nonzero(images.grad) > 0
    camera_grad = model.camera_encoder.projection.weight.grad
    assert camera_grad is not None and torch.isfinite(camera_grad).all()
    assert torch.count_nonzero(camera_grad[:2]) > 0
    assert model.bev_encoder.output[-1].weight.grad is not None


def test_segmentation_is_invariant_to_camera_permutation():
    torch.manual_seed(1)
    model = make_model().eval()
    images, intrinsics, extrinsics = synthetic_inputs()
    with torch.no_grad():
        output = model(images, intrinsics, extrinsics)
        permuted = model(images[:, [1, 0]], intrinsics[:, [1, 0]], extrinsics[:, [1, 0]])
    torch.testing.assert_close(permuted, output, rtol=1e-4, atol=1e-5)


def test_segmentation_handles_multiple_height_bins():
    from lss.models.segmentation import LSSSegmentation

    model = LSSSegmentation(x_bounds=(-8, 8, 1), y_bounds=(-8, 8, 1),
                            z_bounds=(0, 3, 1), depth_bounds=(1, 3, 1),
                            context_channels=8, pretrained=False).eval()
    images, intrinsics, extrinsics = synthetic_inputs()
    with torch.no_grad():
        output = model(images, intrinsics, extrinsics)
    assert model.bev_encoder.in_channels == 24
    assert output.shape == (1, 1, 16, 16)


def test_default_grid_and_depth_bins_match_declared_reference_bounds():
    from lss.models.segmentation import LSSSegmentation

    model = LSSSegmentation(pretrained=False)
    assert model.voxel_pool.grid_size == (200, 200, 1)
    assert model.depth_values.numel() == 41
    assert model.depth_values[0] == 4 and model.depth_values[-1] == 44


@pytest.mark.parametrize("bounds", [(1, 1, 1), (1, 3, 0), (3, 1, 1)])
def test_invalid_depth_bounds_fail(bounds):
    from lss.models.segmentation import LSSSegmentation

    with pytest.raises(ValueError, match="depth"):
        LSSSegmentation(depth_bounds=bounds, pretrained=False)

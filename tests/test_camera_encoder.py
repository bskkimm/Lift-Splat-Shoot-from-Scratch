import pytest
import torch


@pytest.fixture(scope="module")
def encoder():
    from lss.models.camera_encoder import CameraEncoder

    torch.manual_seed(0)
    model = CameraEncoder(depth_bins=5, context_channels=8, pretrained=False)
    model.eval()
    return model


def test_camera_encoder_depth_context_shapes_and_gradients(encoder):
    images = torch.randn(1, 2, 3, 32, 64, requires_grad=True)

    depth, context = encoder(images)

    assert depth.shape == (1, 2, 5, 2, 4)
    assert context.shape == (1, 2, 8, 2, 4)
    torch.testing.assert_close(depth.sum(dim=2), torch.ones(1, 2, 2, 4))
    (depth[:, :, 0].mean() + context.square().mean()).backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert torch.count_nonzero(images.grad) > 0


def test_camera_encoder_is_equivariant_to_camera_order(encoder):
    images = torch.randn(1, 2, 3, 32, 64)
    with torch.no_grad():
        depth, context = encoder(images)
        permuted_depth, permuted_context = encoder(images[:, [1, 0]])

    torch.testing.assert_close(permuted_depth, depth[:, [1, 0]], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(permuted_context, context[:, [1, 0]], rtol=1e-5, atol=1e-6)


def test_camera_encoder_rejects_incompatible_spatial_dimensions(encoder):
    with pytest.raises(ValueError, match="divisible by 32"):
        encoder(torch.randn(1, 1, 3, 31, 64))

import pytest
import torch


def test_voxel_pool_preserves_height_channel_order_and_gradients():
    from lss.models.voxel_pool import VoxelPool

    pool = VoxelPool((0, 2, 1), (0, 1, 1), (0, 2, 1))
    points = torch.tensor([[[[[[0.25, 0.25, 0.25], [0.3, 0.3, 1.1]]]],
                            [[[[0.4, 0.4, 0.4], [2.0, 0.1, 0.1]]]]]])
    features = torch.tensor([1., 2., 3., 4., 5., 6., 100., 200.]).reshape(1, 2, 1, 1, 2, 2)
    features.requires_grad_()

    output = pool(points, features)

    expected = torch.tensor([[[[6.], [0.]], [[8.], [0.]], [[3.], [0.]], [[4.], [0.]]]])
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    expected_grad = torch.tensor([1., 1., 1., 1., 1., 1., 0., 0.]).reshape_as(features)
    torch.testing.assert_close(features.grad, expected_grad)


def test_voxel_pool_xy_axes_and_half_open_bounds():
    from lss.models.voxel_pool import VoxelPool

    pool = VoxelPool((0, 2, 1), (0, 3, 1), (0, 1, 1))
    points = torch.tensor([[[[[[0.1, 2.1, 0.2], [1.1, 0.1, 0.2],
                                [-0.1, 0.1, 0.2], [0.1, 0.1, 1.0]]]]]])
    features = torch.tensor([1., 2., 50., 100.]).reshape(1, 1, 1, 1, 4, 1)

    output = pool(points, features)

    assert output.shape == (1, 1, 2, 3)
    torch.testing.assert_close(output[0, 0], torch.tensor([[0., 0., 1.], [2., 0., 0.]]))


def test_voxel_pool_keeps_batches_separate_and_is_camera_invariant():
    from lss.models.voxel_pool import VoxelPool

    pool = VoxelPool((0, 1, 1), (0, 1, 1), (0, 1, 1))
    points = torch.full((2, 2, 1, 1, 1, 3), 0.5)
    features = torch.tensor([1., 2., 10., 20.]).reshape(2, 2, 1, 1, 1, 1)

    result = pool(points, features)

    torch.testing.assert_close(result[:, 0, 0, 0], torch.tensor([3., 30.]))
    torch.testing.assert_close(pool(points[:, [1, 0]], features[:, [1, 0]]), result)


def test_voxel_pool_empty_valid_set_has_zero_gradients():
    from lss.models.voxel_pool import VoxelPool

    pool = VoxelPool((0, 1, 1), (0, 1, 1), (0, 1, 1))
    points = torch.tensor([[[[[[2., 0., 0.]]]]]])
    features = torch.ones(1, 1, 1, 1, 1, 2, requires_grad=True)

    result = pool(points, features)
    result.sum().backward()

    assert result.sum() == 0
    torch.testing.assert_close(features.grad, torch.zeros_like(features))


def test_voxel_pool_rejects_non_integral_grid():
    from lss.models.voxel_pool import VoxelPool

    with pytest.raises(ValueError, match="integral"):
        VoxelPool((0, 1, 0.3), (0, 1, 1), (0, 1, 1))
    with pytest.raises(ValueError, match="integral"):
        VoxelPool((0, 1e-8, 1), (0, 1, 1), (0, 1, 1))


def test_invalid_point_does_not_contaminate_valid_features_with_nan():
    from lss.models.voxel_pool import VoxelPool

    pool = VoxelPool((0, 1, 1), (0, 1, 1), (0, 1, 1))
    points = torch.tensor([[[[[[0.5, 0.5, 0.5], [2.0, 0.5, 0.5]]]]]])
    features = torch.tensor([1.0, float("nan")]).reshape(1, 1, 1, 1, 2, 1)

    torch.testing.assert_close(pool(points, features), torch.ones(1, 1, 1, 1))


def test_point_just_below_upper_bound_stays_in_last_bin():
    from lss.models.voxel_pool import VoxelPool

    pool = VoxelPool((0, 0.027, 0.009), (0, 1, 1), (0, 1, 1))
    x = torch.nextafter(torch.tensor(0.027), torch.tensor(float("-inf")))
    points = torch.tensor([x, 0.5, 0.5]).reshape(1, 1, 1, 1, 1, 3)
    features = torch.ones(1, 1, 1, 1, 1, 1)

    torch.testing.assert_close(pool(points, features)[0, 0, :, 0], torch.tensor([0., 0., 1.]))


def test_voxel_pool_rejects_integer_coordinates():
    from lss.models.voxel_pool import VoxelPool

    pool = VoxelPool((0, 1, 0.5), (0, 1, 1), (0, 1, 1))
    with pytest.raises(TypeError, match="floating"):
        pool(torch.zeros(1, 1, 1, 1, 1, 3, dtype=torch.int64), torch.ones(1, 1, 1, 1, 1, 1))

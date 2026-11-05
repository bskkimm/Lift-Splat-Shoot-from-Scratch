import torch

from lss.models.lift_splat import LiftSplat


def test_fractional_points_are_assigned_to_half_open_cells():
    pool = LiftSplat((0.0, 2.0, 1.0), (0.0, 1.0, 1.0), (0.0, 1.0, 1.0))
    points = torch.tensor([[[[[[-0.1, 0.2, 0.2], [0.25, 0.2, 0.2],
                                [0.75, 0.2, 0.2], [1.99, 0.2, 0.2],
                                [2.0, 0.2, 0.2]]]]]])
    depth = torch.ones(1, 1, 1, 1, 5)
    context = torch.tensor([[[[[100.0, 2.0, 3.0, 7.0, 100.0]]]]])

    result = pool(depth, context, points)

    torch.testing.assert_allclose(result, torch.tensor([[[[5.0, 7.0]]]]))


def test_fractional_z_bounds_are_checked_before_quantization():
    pool = LiftSplat((0.0, 1.0, 1.0), (0.0, 1.0, 1.0), (0.0, 1.0, 1.0))
    points = torch.tensor([[[[[[0.1, 0.1, -0.01], [0.1, 0.1, 0.99]]]]]])
    depth = torch.ones(1, 1, 1, 1, 2)
    context = torch.tensor([[[[[100.0, 3.0]]]]])

    torch.testing.assert_allclose(pool(depth, context, points), torch.tensor([[[[3.0]]]]))


def test_pooling_accumulates_cameras_per_batch_and_backpropagates():
    pool = LiftSplat((-1.0, 1.0, 0.5), (0.0, 1.0, 1.0), (0.0, 1.0, 1.0))
    points = torch.tensor([[[[[[-0.75, 0.1, 0.1]]]], [[[[0.75, 0.1, 0.1]]]]],
                           [[[[[-0.75, 0.1, 0.1]]]], [[[[0.75, 0.1, 0.1]]]]]])
    depth = torch.ones(2, 2, 1, 1, 1, requires_grad=True)
    context = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2, 1, 1, 1).requires_grad_()

    result = pool(depth, context, points)

    expected = torch.tensor([[[[1.0, 0.0, 0.0, 2.0]]],
                             [[[3.0, 0.0, 0.0, 4.0]]]])
    torch.testing.assert_allclose(result, expected)
    result.sum().backward()
    torch.testing.assert_allclose(context.grad, torch.ones_like(context))
    torch.testing.assert_allclose(depth.grad, context.detach())

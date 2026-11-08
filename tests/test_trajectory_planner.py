import pytest
import torch


def test_metric_trajectories_convert_to_xy_grid_indices():
    from lss.models.planner import metric_trajectory_cells

    trajectories = torch.tensor([[[[-2., -1.], [-0.1, 0.9]], [[0.1, -0.1], [1.99, 1.99]]]])
    result = metric_trajectory_cells(trajectories, (-2, 2, 1), (-1, 2, 1))
    assert result.tolist() == [[[[0, 0], [1, 1]], [[2, 0], [3, 2]]]]


@pytest.mark.parametrize("point", [[2., 0.], [-2.01, 0.], [0., float("nan")]])
def test_metric_trajectories_reject_invalid_points(point):
    from lss.models.planner import metric_trajectory_cells

    with pytest.raises(ValueError):
        metric_trajectory_cells(torch.tensor(point).reshape(1, 1, 1, 2), (-2, 2, 1), (-1, 2, 1))


def test_planner_forward_and_loss_backpropagate_through_cost_map():
    from lss.models.planner import LSSPlanner
    from lss.models.trajectory import trajectory_classification_loss

    torch.manual_seed(2)
    planner = LSSPlanner(x_bounds=(-8, 8, 1), y_bounds=(-8, 8, 1), z_bounds=(0, 3, 3),
                         depth_bounds=(1, 3, 1), context_channels=8, pretrained=False)
    images = torch.randn(1, 2, 3, 32, 64)
    intrinsics = torch.tensor([[10., 0., 32.], [0., 10., 16.], [0., 0., 1.]])[None, None].expand(1, 2, 3, 3).clone()
    extrinsics = torch.eye(4)[None, None].expand(1, 2, 4, 4).clone()
    templates = torch.tensor([[[[-1., 0.], [0., 0.]], [[1., 0.], [2., 0.]]]])

    output = planner(images, intrinsics, extrinsics, templates)
    loss = trajectory_classification_loss(output["logits"], templates, templates[:, 0])
    loss.backward()

    assert output["cost_map"].shape == (1, 16, 16)
    torch.testing.assert_close(output["probabilities"].sum(dim=1), torch.ones(1))
    assert planner.cost_model.bev_encoder.output[-1].weight.grad is not None
    assert torch.count_nonzero(planner.cost_model.camera_encoder.projection.weight.grad) > 0

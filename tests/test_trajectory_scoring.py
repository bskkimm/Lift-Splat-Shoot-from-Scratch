import pytest
import torch


def test_trajectory_logits_sum_costs_and_softmax_with_batch_isolation():
    from lss.models.trajectory import trajectory_logits

    cost = torch.tensor([[[1., 2.], [3., 4.]], [[10., 20.], [30., 40.]]])
    cells = torch.tensor([[[[0, 0], [1, 1]], [[0, 1], [1, 0]]],
                          [[[0, 0], [0, 0]], [[1, 1], [0, 1]]]])
    logits = trajectory_logits(cost, cells)
    torch.testing.assert_close(logits, torch.tensor([[-5., -5.], [-20., -60.]]))
    torch.testing.assert_close(logits[0].softmax(0), torch.tensor([0.5, 0.5]))
    assert logits[1].softmax(0)[0] > 0.999


def test_trajectory_logits_accumulate_gradients_for_shared_cells():
    from lss.models.trajectory import trajectory_logits

    cost = torch.zeros(1, 2, 2, requires_grad=True)
    cells = torch.tensor([[[[0, 0], [1, 1]], [[0, 0], [0, 0]]]])
    trajectory_logits(cost, cells).sum().backward()
    torch.testing.assert_close(cost.grad, torch.tensor([[[-3., 0.], [0., -1.]]]))


def test_nearest_template_and_classification_loss_have_expected_gradients():
    from lss.models.trajectory import nearest_template, trajectory_classification_loss

    templates = torch.tensor([[[[0., 0.], [1., 0.]], [[0., 0.], [3., 0.]]]])
    expert = torch.tensor([[[0., 0.], [2.8, 0.]]])
    assert nearest_template(templates, expert).tolist() == [1]
    logits = torch.zeros(1, 2, requires_grad=True)
    loss = trajectory_classification_loss(logits, templates, expert)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.69314718))
    torch.testing.assert_close(logits.grad, torch.tensor([[0.5, -0.5]]))


@pytest.mark.parametrize("cells", [
    torch.tensor([[[[-1, 0]]]]), torch.tensor([[[[2, 0]]]]),
    torch.tensor([[[[0., 0.]]]]),
])
def test_trajectory_logits_reject_invalid_indices(cells):
    from lss.models.trajectory import trajectory_logits

    with pytest.raises((ValueError, TypeError)):
        trajectory_logits(torch.zeros(1, 2, 2), cells)


def test_nearest_template_rejects_shape_mismatch():
    from lss.models.trajectory import nearest_template

    with pytest.raises(ValueError):
        nearest_template(torch.zeros(1, 2, 3, 2), torch.zeros(1, 2, 2))


def test_shoot_classification_backpropagates_into_cost_map():
    from lss.models.trajectory import trajectory_classification_loss, trajectory_logits

    cost = torch.zeros(1, 2, 2, requires_grad=True)
    cells = torch.tensor([[[[0, 0], [0, 1]], [[1, 0], [1, 1]]]])
    templates = cells.float()
    expert = templates[:, 1].clone()
    loss = trajectory_classification_loss(trajectory_logits(cost, cells), templates, expert)
    loss.backward()
    torch.testing.assert_close(cost.grad, torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]]))

import math

import pytest
import torch

from lss import losses


def test_weighted_segmentation_bce_and_gradients_at_zero_logits():
    logits = torch.zeros(2, requires_grad=True)
    targets = torch.tensor([1.0, 0.0])

    loss = losses.binary_segmentation_loss(logits, targets, pos_weight=2.0)

    torch.testing.assert_allclose(loss, torch.tensor(1.5 * math.log(2.0)))
    loss.backward()
    torch.testing.assert_allclose(logits.grad, torch.tensor([-0.5, 0.25]))


@pytest.mark.parametrize("logits,targets", [
    (torch.zeros(2), torch.zeros(1)),
    (torch.empty(0), torch.empty(0)),
])
def test_segmentation_bce_rejects_mismatched_or_empty_inputs(logits, targets):
    with pytest.raises(ValueError):
        losses.binary_segmentation_loss(logits, targets)


def test_segmentation_iou_uses_zero_logit_threshold_and_aggregate_counts():
    from lss.metrics import segmentation_intersection_union, segmentation_iou

    first = segmentation_intersection_union(torch.tensor([2., -2., 0.]), torch.tensor([1., 1., 0.]))
    second = segmentation_intersection_union(torch.tensor([2.]), torch.tensor([0.]))

    assert first == (1, 2)
    assert second == (0, 1)
    assert segmentation_iou(first[0] + second[0], first[1] + second[1]) == pytest.approx(1 / 3)


def test_empty_union_matches_reference_batch_convention():
    from lss.metrics import segmentation_intersection_union, segmentation_iou

    counts = segmentation_intersection_union(torch.tensor([-1.]), torch.tensor([0.]))
    assert counts == (0, 0)
    assert segmentation_iou(*counts) == 1.0

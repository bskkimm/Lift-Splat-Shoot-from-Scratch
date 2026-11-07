import math

import pytest
import torch

from tests.test_segmentation_engine import TinySegmentationModel, tiny_batch


def test_validation_epoch_weights_loss_and_aggregates_pixel_counts():
    from lss.segmentation_engine import run_segmentation_epoch

    model = TinySegmentationModel().train()
    first = tiny_batch()
    second = {key: value[:1] for key, value in tiny_batch().items()}
    second["bev_target"] = torch.zeros_like(second["bev_target"])

    result = run_segmentation_epoch(model, [first, second], device="cpu", pos_weight=2.0)

    assert result.samples == 3
    assert result.loss == pytest.approx(5 * math.log(2) / 3)
    assert (result.intersection, result.union, result.iou) == (0, 8, 0.0)
    assert model.training
    assert model.forward_modes == [(False, False), (False, False)]


def test_training_epoch_performs_optimizer_updates():
    from lss.segmentation_engine import run_segmentation_epoch

    model = TinySegmentationModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    before = model.head.weight.detach().clone()

    result = run_segmentation_epoch(model, [tiny_batch(), tiny_batch()],
                                    device="cpu", optimizer=optimizer)

    assert result.samples == 4
    assert not torch.equal(model.head.weight, before)
    assert model.forward_modes == [(True, True), (True, True)]


def test_empty_epoch_does_not_produce_a_metric():
    from lss.segmentation_engine import run_segmentation_epoch

    with pytest.raises(ValueError, match="empty"):
        run_segmentation_epoch(TinySegmentationModel(), [], device="cpu")


def test_validation_rejects_gradient_clipping():
    from lss.segmentation_engine import run_segmentation_epoch

    with pytest.raises(ValueError, match="gradient clipping"):
        run_segmentation_epoch(TinySegmentationModel(), [tiny_batch()], device="cpu", grad_clip_norm=1.0)


def test_epoch_iou_is_undefined_for_empty_union():
    from lss.segmentation_engine import run_segmentation_epoch

    batch = tiny_batch()
    batch["bev_target"] = torch.zeros_like(batch["bev_target"])
    result = run_segmentation_epoch(TinySegmentationModel(), [batch], device="cpu")
    assert (result.intersection, result.union, result.iou) == (0, 0, None)

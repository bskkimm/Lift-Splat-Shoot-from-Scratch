import torch
import pytest
from torch import nn

from lss.data.collate import collate_camera_samples
from lss.data.nuscenes_dataset import NuScenesCameraDataset
from lss.models.segmentation import LSSSegmentation
from tests.test_nuscenes_reference_frames import make_reference_fixture


class TinySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(3, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.forward_modes = []

    def forward(self, images, intrinsics, extrinsics, post_rots=None, post_trans=None):
        self.forward_modes.append((self.training, torch.is_grad_enabled()))
        return self.head(images[:, 0])


def tiny_batch():
    return {
        "images": torch.ones(2, 1, 3, 2, 2),
        "intrinsics": torch.eye(3)[None, None].expand(2, 1, 3, 3),
        "extrinsics": torch.eye(4)[None, None].expand(2, 1, 4, 4),
        "bev_target": torch.ones(2, 1, 2, 2),
    }


def test_training_step_updates_parameters_and_reduces_loss():
    from lss.segmentation_engine import train_segmentation_step

    model = TinySegmentationModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    initial = model.head.weight.detach().clone()
    first = train_segmentation_step(model, tiny_batch(), optimizer, device="cpu")
    last = first
    for _ in range(4):
        last = train_segmentation_step(model, tiny_batch(), optimizer, device="cpu")

    assert first.samples == 2 and (first.intersection, first.union) == (0, 8)
    assert last.loss < first.loss
    assert not torch.equal(initial, model.head.weight)
    assert model.forward_modes == [(True, True)] * 5


def test_evaluation_step_disables_gradients_and_restores_training_mode():
    from lss.segmentation_engine import evaluate_segmentation_step

    model = TinySegmentationModel().train()
    result = evaluate_segmentation_step(model, tiny_batch(), device="cpu")

    assert model.forward_modes == [(False, False)]
    assert model.training
    assert model.head.weight.grad is None
    assert result.samples == 2 and (result.intersection, result.union) == (0, 8)


def test_evaluation_restores_mixed_submodule_modes():
    from lss.segmentation_engine import evaluate_segmentation_step

    model = TinySegmentationModel().train()
    model.head.eval()
    evaluate_segmentation_step(model, tiny_batch(), device="cpu")
    assert model.training
    assert not model.head.training


def test_evaluation_restores_modes_after_invalid_batch():
    from lss.segmentation_engine import evaluate_segmentation_step

    model = TinySegmentationModel().train()
    model.head.eval()
    batch = tiny_batch()
    del batch["bev_target"]
    with pytest.raises(KeyError, match="bev_target"):
        evaluate_segmentation_step(model, batch, device="cpu")
    assert model.training and not model.head.training


def test_raw_synthetic_batch_updates_assembled_segmentation_model(tmp_path):
    from lss.segmentation_engine import train_segmentation_step

    make_reference_fixture(tmp_path)
    dataset = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini",
                                    image_size=(32, 64),
                                    bev_bounds=((0, 8, 1), (-4, 4, 1)))
    sample = dataset[0]
    batch = collate_camera_samples([sample, sample])
    model = LSSSegmentation(x_bounds=(0, 8, 1), y_bounds=(-4, 4, 1),
                            z_bounds=(-5, 5, 10), depth_bounds=(1, 3, 1),
                            context_channels=8, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.bev_encoder.output[-1].weight.detach().clone()

    result = train_segmentation_step(model, batch, optimizer, device="cpu")

    assert result.samples == 2
    assert result.loss > 0
    assert not torch.equal(before, model.bev_encoder.output[-1].weight)

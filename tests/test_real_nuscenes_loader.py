"""Opt-in data-only nuScenes loader verification; no model training."""

import os

import pytest
import torch

from lss.data.nuscenes_dataset import NuScenesCameraDataset


@pytest.mark.skipif(not os.environ.get("LSS_REAL_NUSCENES_ROOT"),
                    reason="set LSS_REAL_NUSCENES_ROOT to run real-data loader smoke")
def test_real_nuscenes_validation_sample_and_official_split_count():
    root = os.environ["LSS_REAL_NUSCENES_ROOT"]
    dataset = NuScenesCameraDataset(dataroot=root, version="v1.0-trainval", split="val",
                                    image_size=(128, 352),
                                    bev_bounds=((-50, 50, 0.5), (-50, 50, 0.5)))
    assert len(dataset) == 6019
    sample = dataset[0]
    assert sample["images"].shape == (6, 3, 128, 352)
    assert sample["intrinsics"].shape == (6, 3, 3)
    assert sample["extrinsics"].shape == (6, 4, 4)
    assert sample["bev_target"].shape == (1, 200, 200)
    assert torch.isfinite(sample["images"]).all()
    assert torch.isfinite(sample["intrinsics"]).all()
    assert torch.isfinite(sample["extrinsics"]).all()
    assert torch.isin(sample["bev_target"], torch.tensor([0., 1.])).all()

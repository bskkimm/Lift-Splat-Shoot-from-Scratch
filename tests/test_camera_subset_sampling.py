import numpy as np
import pytest
import torch
from PIL import Image

from lss.data.nuscenes_dataset import NuScenesCameraDataset


def camera_record(tmp_path, count=6):
    paths = []
    for index in range(count):
        path = tmp_path / f"camera-{index}.png"
        Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(path)
        paths.append(str(path))
    return {"image_paths": paths,
            "intrinsics": [(torch.eye(3) * (index + 1)).tolist() for index in range(count)],
            "extrinsics": [(torch.eye(4) + torch.diag(torch.tensor([0., 0., 0., float(index)]))).tolist()
                           for index in range(count)]}


def test_training_camera_subset_is_seeded_unique_and_aligned(tmp_path):
    record = camera_record(tmp_path)
    dataset = NuScenesCameraDataset([record], camera_count=3, training=True)
    np.random.seed(8)
    item = dataset[0]
    np.random.seed(8)
    repeated = dataset[0]
    assert item["images"].shape[0] == 3
    torch.testing.assert_close(item["intrinsics"], repeated["intrinsics"])
    selected = item["intrinsics"][:, 0, 0]
    assert len(selected.unique()) == 3
    torch.testing.assert_close(item["extrinsics"][:, 3, 3], selected)
    expected_red = ((selected - 1) * 20 / 255 - 0.485) / 0.229
    torch.testing.assert_close(item["images"][:, 0, 0, 0], expected_red)


def test_validation_keeps_all_cameras_even_if_training_count_is_smaller(tmp_path):
    dataset = NuScenesCameraDataset([camera_record(tmp_path)], camera_count=3, training=False)
    item = dataset[0]
    assert item["images"].shape[0] == 6
    torch.testing.assert_close(item["intrinsics"][:, 0, 0], torch.arange(1., 7.))


def test_fixed_augmentation_parameters_follow_selected_camera(tmp_path):
    record = camera_record(tmp_path)
    record["augmentations"] = [
        {"resize": 1., "resize_dims": (8, 8), "crop": (i, 0, i + 4, 4),
         "flip": False, "rotate": 0.} for i in range(6)
    ]
    dataset = NuScenesCameraDataset([record], camera_count=3, training=True)
    np.random.seed(42)
    item = dataset[0]
    torch.testing.assert_close(item["post_trans"][:, 0], 1. - item["intrinsics"][:, 0, 0])


def test_mismatched_camera_calibration_counts_are_rejected(tmp_path):
    record = camera_record(tmp_path)
    record["extrinsics"].pop()
    with pytest.raises(ValueError, match="calibration"):
        NuScenesCameraDataset([record], camera_count=3, training=True)[0]


@pytest.mark.parametrize("count", [0, -1, 7])
def test_invalid_training_camera_count_is_rejected(tmp_path, count):
    with pytest.raises(ValueError, match="camera_count"):
        NuScenesCameraDataset([camera_record(tmp_path)], camera_count=count, training=True)[0]

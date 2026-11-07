import pytest
import torch
from PIL import Image

from lss.data.collate import collate_camera_samples
from lss.data.nuscenes_dataset import NuScenesCameraDataset
from lss.geometry import camera_to_ego


def test_dataset_and_collate_preserve_intrinsics_and_post_transform(tmp_path):
    image = tmp_path / "camera.png"
    Image.new("RGB", (10, 8)).save(image)
    record = {
        "image_paths": [str(image)],
        "intrinsics": [[[2, 0, 0], [0, 2, 0], [0, 0, 1]]],
        "extrinsics": [torch.eye(4).tolist()],
        "augmentations": [{"resize": 2., "resize_dims": (20, 16),
                           "crop": (2, 3, 10, 9), "flip": True, "rotate": 0.}],
    }
    item = NuScenesCameraDataset([record])[0]
    batch = collate_camera_samples([item, item])
    assert batch["images"].shape == (2, 1, 3, 6, 8)
    torch.testing.assert_close(batch["intrinsics"][0, 0], torch.diag(torch.tensor([2., 2., 1.])))
    original = torch.tensor([1., 2., 3.])
    augmented = batch["post_rots"][0, 0] @ original + batch["post_trans"][0, 0]
    result = camera_to_ego(augmented.reshape(1, 1, 1, 1, 1, 3), batch["intrinsics"][:1],
                           batch["extrinsics"][:1], batch["post_rots"][:1], batch["post_trans"][:1])
    torch.testing.assert_close(result.flatten(), torch.tensor([1.5, 3., 3.]))


def test_dataset_rejects_incomplete_camera_augmentation_records(tmp_path):
    image = tmp_path / "camera.png"
    Image.new("RGB", (10, 8)).save(image)
    record = {"image_paths": [str(image)], "intrinsics": [torch.eye(3).tolist()],
              "extrinsics": [torch.eye(4).tolist()], "augmentations": []}
    with pytest.raises(ValueError, match="augmentation"):
        NuScenesCameraDataset([record])[0]


def test_collate_rejects_mixed_post_transform_presence():
    sample = {"images": torch.zeros(1, 3, 2, 2), "intrinsics": torch.eye(3)[None],
              "extrinsics": torch.eye(4)[None], "boxes": torch.zeros(0, 9),
              "labels": torch.zeros(0, dtype=torch.long)}
    augmented = {**sample, "post_rots": torch.eye(3)[None], "post_trans": torch.zeros(1, 3)}
    with pytest.raises(ValueError, match="post"):
        collate_camera_samples([sample, augmented])


def test_dataset_rejects_double_resizing(tmp_path):
    image = tmp_path / "camera.png"
    Image.new("RGB", (10, 8)).save(image)
    record = {"image_paths": [str(image)], "intrinsics": [torch.eye(3).tolist()],
              "extrinsics": [torch.eye(4).tolist()],
              "augmentations": [{"resize": 1., "resize_dims": (10, 8),
                                 "crop": (0, 0, 8, 6), "flip": False, "rotate": 0.}]}
    with pytest.raises(ValueError, match="image_size"):
        NuScenesCameraDataset([record], image_size=(6, 8))[0]


def test_collate_rejects_unpaired_post_transform():
    sample = {"images": torch.zeros(1, 3, 2, 2), "intrinsics": torch.eye(3)[None],
              "extrinsics": torch.eye(4)[None], "boxes": torch.zeros(0, 9),
              "labels": torch.zeros(0, dtype=torch.long), "post_rots": torch.eye(3)[None]}
    with pytest.raises(ValueError, match="paired"):
        collate_camera_samples([sample])

import numpy as np
import pytest
import torch
from PIL import Image

from lss.data.collate import collate_camera_samples
from lss.data.nuscenes_dataset import NuScenesCameraDataset


def test_validation_sampling_is_deterministic_and_matches_reference_crop():
    from lss.data.image_augmentation import ImageAugmentation

    policy = ImageAugmentation(final_dim=(4, 8), resize_limits=(0.8, 1.2),
                               bottom_limits=(0.0, 0.2), rotation_limits=(-5., 5.),
                               random_flip=True, training=False)
    image = Image.new("RGB", (16, 8))
    first, rotation, translation = policy(image)
    second, rotation2, translation2 = policy(image)
    assert first.size == second.size == (8, 4)
    torch.testing.assert_close(rotation, rotation2)
    torch.testing.assert_close(translation, translation2)
    # resize=.5: 8x4; bottom crop starts at -1, and horizontal crop starts at zero.
    torch.testing.assert_close(rotation, torch.diag(torch.tensor([0.5, 0.5, 1.])))
    torch.testing.assert_close(translation, torch.tensor([0., 1., 0.]))


def test_training_sampling_is_repeatable_when_numpy_is_seeded():
    from lss.data.image_augmentation import ImageAugmentation

    policy = ImageAugmentation(final_dim=(4, 8), resize_limits=(0.6, 0.9),
                               bottom_limits=(0., 0.2), rotation_limits=(-5., 5.),
                               random_flip=True, training=True)
    image = Image.new("RGB", (16, 8))
    np.random.seed(17)
    _, first_rotation, first_translation = policy(image)
    _, next_rotation, next_translation = policy(image)
    np.random.seed(17)
    _, repeated_rotation, repeated_translation = policy(image)
    torch.testing.assert_close(first_rotation, repeated_rotation)
    torch.testing.assert_close(first_translation, repeated_translation)
    assert not torch.equal(first_rotation, next_rotation) or not torch.equal(first_translation, next_translation)


def test_dataset_augmentation_policy_returns_batched_post_transforms(tmp_path):
    from lss.data.image_augmentation import ImageAugmentation

    image = tmp_path / "camera.png"
    Image.new("RGB", (16, 8)).save(image)
    record = {"image_paths": [str(image)], "intrinsics": [torch.eye(3).tolist()],
              "extrinsics": [torch.eye(4).tolist()]}
    policy = ImageAugmentation(final_dim=(4, 8), resize_limits=(0.8, 1.2),
                               bottom_limits=(0., 0.2), rotation_limits=(-5., 5.),
                               random_flip=True, training=False)
    item = NuScenesCameraDataset([record], augmentation=policy)[0]
    batch = collate_camera_samples([item])
    assert batch["images"].shape == (1, 1, 3, 4, 8)
    torch.testing.assert_close(batch["intrinsics"][0, 0], torch.eye(3))
    torch.testing.assert_close(batch["post_trans"][0, 0], torch.tensor([0., 1., 0.]))


def test_dataset_rejects_policy_plus_resize():
    from lss.data.image_augmentation import ImageAugmentation

    policy = ImageAugmentation(final_dim=(4, 8), resize_limits=(0.8, 1.2),
                               bottom_limits=(0., 0.2), rotation_limits=(-5., 5.),
                               random_flip=True, training=True)
    with pytest.raises(ValueError, match="image_size"):
        NuScenesCameraDataset([], image_size=(4, 8), augmentation=policy)


def test_raw_fixture_augmentation_runs_through_segmentation_model(tmp_path):
    from lss.data.image_augmentation import ImageAugmentation
    from lss.models.segmentation import LSSSegmentation
    from tests.test_nuscenes_reference_frames import make_reference_fixture

    make_reference_fixture(tmp_path)
    policy = ImageAugmentation(final_dim=(32, 64), resize_limits=(32., 32.),
                               bottom_limits=(0., 0.), rotation_limits=(0., 0.),
                               random_flip=False, training=False)
    dataset = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini",
                                    bev_bounds=((-8, 8, 1), (-8, 8, 1)), augmentation=policy)
    batch = collate_camera_samples([dataset[0]])
    model = LSSSegmentation(x_bounds=(-8, 8, 1), y_bounds=(-8, 8, 1),
                            z_bounds=(-5, 5, 10), depth_bounds=(1, 3, 1),
                            context_channels=8, pretrained=False).eval()
    with torch.no_grad():
        output = model(batch["images"], batch["intrinsics"], batch["extrinsics"],
                       batch["post_rots"], batch["post_trans"])
    assert output.shape == batch["bev_target"].shape == (1, 1, 16, 16)
    assert torch.isfinite(output).all()

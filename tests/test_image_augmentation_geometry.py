import pytest
import torch
from PIL import Image

from lss.geometry import camera_to_ego


def test_resize_crop_flip_affine_matches_reference_convention():
    from lss.data.image_augmentation import augment_image

    image = Image.new("RGB", (10, 8))
    transformed, rotation, translation = augment_image(
        image, resize=2.0, resize_dims=(20, 16), crop=(2, 3, 10, 9), flip=True, rotate=0.0,
    )
    assert transformed.size == (8, 6)
    torch.testing.assert_close(rotation, torch.diag(torch.tensor([-2., 2., 1.])))
    torch.testing.assert_close(translation, torch.tensor([10., -3., 0.]))


def test_rotation_affine_round_trips_through_unprojection():
    from lss.data.image_augmentation import augment_image

    _, rotation, translation = augment_image(Image.new("RGB", (8, 6)), resize=1.0,
                                             resize_dims=(8, 6), crop=(0, 0, 8, 6),
                                             flip=False, rotate=90.0)
    original = torch.tensor([2., 1., 3.])
    augmented = rotation @ original + translation
    torch.testing.assert_close(augmented, torch.tensor([2., 5., 3.]), atol=1e-5, rtol=1e-5)
    actual = camera_to_ego(augmented.reshape(1, 1, 1, 1, 1, 3), torch.eye(3)[None, None],
                           torch.eye(4)[None, None], rotation[None, None], translation[None, None])
    torch.testing.assert_close(actual.flatten(), torch.tensor([6., 3., 3.]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("resize,crop", [(0, (0, 0, 8, 6)), (1, (0, 0, 0, 6))])
def test_invalid_augmentation_arguments_are_rejected(resize, crop):
    from lss.data.image_augmentation import augment_image

    with pytest.raises(ValueError):
        augment_image(Image.new("RGB", (8, 6)), resize=resize, resize_dims=(8, 6),
                      crop=crop, flip=False, rotate=0.0)

"""Deterministic image augmentation and its LSS post-transform geometry."""

import math

import numpy as np
import torch
from PIL import Image


class ImageAugmentation:
    """Sample reference-style per-camera image augmentations.

    Training draws from NumPy's global RNG. Validation is deterministic.
    ``final_dim`` is (height, width); image sizes use PIL's (width, height).
    """

    def __init__(self, final_dim, resize_limits, bottom_limits, rotation_limits,
                 random_flip, training):
        if len(final_dim) != 2 or any(not isinstance(value, int) or value <= 0 for value in final_dim):
            raise ValueError("final_dim must contain positive integer height and width")
        for name, limits in (("resize", resize_limits), ("bottom", bottom_limits),
                             ("rotation", rotation_limits)):
            if len(limits) != 2 or not all(math.isfinite(v) for v in limits) or limits[0] > limits[1]:
                raise ValueError(f"invalid {name} limits")
        if resize_limits[0] <= 0 or bottom_limits[0] < 0 or bottom_limits[1] > 1:
            raise ValueError("resize must be positive and bottom limits must be within [0,1]")
        self.final_dim = tuple(final_dim)
        self.resize_limits = tuple(resize_limits)
        self.bottom_limits = tuple(bottom_limits)
        self.rotation_limits = tuple(rotation_limits)
        self.random_flip = random_flip
        self.training = training

    def __call__(self, image):
        width, height = image.size
        final_height, final_width = self.final_dim
        if self.training:
            resize = float(np.random.uniform(*self.resize_limits))
        else:
            resize = max(final_height / height, final_width / width)
        new_width, new_height = int(width * resize), int(height * resize)
        if self.training:
            bottom = float(np.random.uniform(*self.bottom_limits))
            left = int(np.random.uniform(0, max(0, new_width - final_width)))
            flip = bool(self.random_flip and np.random.choice([0, 1]))
            rotate = float(np.random.uniform(*self.rotation_limits))
        else:
            bottom = sum(self.bottom_limits) / 2
            left = int(max(0, new_width - final_width) / 2)
            flip = False
            rotate = 0.0
        top = int((1 - bottom) * new_height) - final_height
        return augment_image(image, resize, (new_width, new_height),
                             (left, top, left + final_width, top + final_height),
                             flip, rotate)


def augment_image(image, resize, resize_dims, crop, flip, rotate):
    """Apply resize/crop/flip/rotation and return image, post-rotation, post-translation.

    ``resize_dims`` is (width, height), ``crop`` is (left, top, right, bottom),
    and ``rotate`` is degrees. The returned [3,3] and [3] tensors map original
    pixel/depth coordinates to augmented pixel/depth coordinates. This uses the
    reference implementation's continuous-width flip convention.
    """
    left, top, right, bottom = crop
    if (not math.isfinite(resize) or resize <= 0 or not math.isfinite(rotate)
            or len(resize_dims) != 2 or min(resize_dims) <= 0
            or right <= left or bottom <= top):
        raise ValueError("invalid image augmentation parameters")

    image = image.resize(resize_dims).crop(crop)
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = image.rotate(rotate)

    post_rot = torch.eye(2) * resize
    post_trans = torch.tensor([-float(left), -float(top)])
    width, height = right - left, bottom - top
    if flip:
        reflection = torch.tensor([[-1., 0.], [0., 1.]])
        post_rot = reflection @ post_rot
        post_trans = reflection @ post_trans + torch.tensor([float(width), 0.])
    angle = math.radians(rotate)
    rotation = torch.tensor([[math.cos(angle), math.sin(angle)],
                             [-math.sin(angle), math.cos(angle)]])
    center = torch.tensor([width / 2., height / 2.])
    post_rot = rotation @ post_rot
    post_trans = rotation @ (post_trans - center) + center
    result_rotation = torch.eye(3)
    result_translation = torch.zeros(3)
    result_rotation[:2, :2] = post_rot
    result_translation[:2] = post_trans
    return image, result_rotation, result_translation

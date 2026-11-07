"""Deterministic image augmentation and its LSS post-transform geometry."""

import math

import torch
from PIL import Image


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

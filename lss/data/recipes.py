"""Shared dataset configuration for named practical experiments."""

from .image_augmentation import ImageAugmentation


def dataset_options(config, training):
    """Return dataset options for the checkpoint's named recipe and mode."""
    recipe = config["recipe"]
    if recipe == "practical-resize-fp32-v1":
        return {"image_size": tuple(config["image_size"])}
    if recipe == "reference-data-fp32-v1":
        augmentation = config["augmentation"]
        return {
            "augmentation": ImageAugmentation(
                final_dim=tuple(config["image_size"]),
                resize_limits=tuple(augmentation["resize_limits"]),
                bottom_limits=tuple(augmentation["bottom_limits"]),
                rotation_limits=tuple(augmentation["rotation_limits"]),
                random_flip=augmentation["random_flip"], training=training,
            ),
            "camera_count": config["camera_count"],
            "training": training,
        }
    raise ValueError(f"unsupported dataset recipe: {recipe}")

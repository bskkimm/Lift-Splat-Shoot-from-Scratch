import torch


def normalize_images(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    shape = (1, 1, 3, 1, 1) if images.ndim == 5 else (1, 3, 1, 1)
    mean = torch.tensor(mean, device=images.device, dtype=images.dtype).view(*shape)
    std = torch.tensor(std, device=images.device, dtype=images.dtype).view(*shape)
    return (images - mean) / std

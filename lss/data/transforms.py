import torch


def normalize_images(images, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean, device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor(std, device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    return (images - mean) / std

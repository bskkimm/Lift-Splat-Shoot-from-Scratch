from dataclasses import dataclass
import torch


@dataclass
class CameraBatch:
    images: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor

    def __post_init__(self):
        if not self.images.is_floating_point() or not self.intrinsics.is_floating_point() or not self.extrinsics.is_floating_point(): raise TypeError("camera tensors must use floating point dtype")
        if self.images.ndim != 5:
            raise ValueError("images must be [B, C, 3, H, W]")
        b, c = self.images.shape[:2]
        if self.intrinsics.shape != (b, c, 3, 3):
            raise ValueError("intrinsics must be [B, C, 3, 3]")
        if self.extrinsics.shape != (b, c, 4, 4):
            raise ValueError("extrinsics must be [B, C, 4, 4]")


@dataclass
class Sample:
    cameras: CameraBatch
    boxes: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self):
        if self.boxes.ndim != 2 or self.boxes.shape[0] != self.labels.shape[0]: raise ValueError("boxes and labels must have matching leading dimensions")

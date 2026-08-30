from dataclasses import dataclass
import torch


@dataclass
class CameraBatch:
    images: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor

    def __post_init__(self):
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

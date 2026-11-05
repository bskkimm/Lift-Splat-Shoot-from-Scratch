"""Composable multi-camera lift-splat BEV segmentation model."""

import math

import torch
from torch import nn

from lss.geometry import camera_to_ego, make_frustum
from .bev_encoder import BEVEncoder
from .camera_encoder import CameraEncoder
from .voxel_pool import VoxelPool


class LSSSegmentation(nn.Module):
    """Predict BEV logits from calibrated, preprocessed multi-camera images.

    Images are [B,N,3,H,W], intrinsics [B,N,3,3] and camera-to-ego
    transforms [B,N,4,4]. Optional post-transform pairs describe image
    augmentation relative to the provided intrinsics. The output is raw
    segmentation logits [B,K,X,Y]. Pretrained image weights are opt-in.
    """

    def __init__(self, x_bounds=(-50.0, 50.0, 0.5), y_bounds=(-50.0, 50.0, 0.5),
                 z_bounds=(-10.0, 10.0, 20.0), depth_bounds=(4.0, 45.0, 1.0),
                 context_channels=64, out_channels=1, pretrained=False):
        super().__init__()
        start, stop, step = depth_bounds
        if not all(math.isfinite(value) for value in depth_bounds) or start <= 0 or stop <= start or step <= 0:
            raise ValueError("depth bounds must be finite, positive and increasing")
        depth_values = torch.arange(start, stop, step, dtype=torch.float32)
        if depth_values.numel() == 0:
            raise ValueError("depth bounds must produce at least one depth bin")
        self.register_buffer("depth_values", depth_values)
        self.camera_encoder = CameraEncoder(depth_values.numel(), context_channels, pretrained=pretrained)
        self.voxel_pool = VoxelPool(x_bounds, y_bounds, z_bounds)
        self.bev_encoder = BEVEncoder(context_channels * self.voxel_pool.grid_size[2], out_channels)

    def forward(self, images, intrinsics, extrinsics, post_rots=None, post_trans=None):
        depth, context = self.camera_encoder(images)
        frustum = make_frustum(depth.shape[-2], depth.shape[-1], self.depth_values,
                               device=images.device, image_size=images.shape[-2:])
        points = frustum[None, None].expand(images.shape[0], images.shape[1], -1, -1, -1, -1)
        points = camera_to_ego(points, intrinsics, extrinsics,
                               post_rots=post_rots, post_trans=post_trans)
        lifted = depth.unsqueeze(-1) * context.permute(0, 1, 3, 4, 2).unsqueeze(2)
        return self.bev_encoder(self.voxel_pool(points, lifted))

import torch
import torch.nn as nn
from .backbone import ImageBackbone
from .depth import DepthContextNet
from .lift_splat import LiftSplat
from lss.geometry import camera_to_ego, make_frustum


class LSS(nn.Module):
    def __init__(self, depth_bins=8, feature_channels=64, bev_bounds=((-8., 8., 1.), (-8., 8., 1.), (-2., 2., 1.))):
        super().__init__()
        if depth_bins <= 0 or feature_channels <= 0: raise ValueError("LSS dimensions must be positive")
        self.backbone = ImageBackbone(channels=feature_channels)
        self.depth = DepthContextNet(feature_channels, depth_bins, feature_channels)
        self.lift_splat = LiftSplat(*bev_bounds)

    def forward(self, images, intrinsics, extrinsics, depths, post_rots=None, post_trans=None):
        features = self.backbone(images)
        depth, context = self.depth(features)
        frustum = make_frustum(features.shape[-2], features.shape[-1], depths,
                               images.device, image_size=images.shape[-2:])
        points = camera_to_ego(frustum[None, None].expand(images.shape[0], images.shape[1], -1, -1, -1, -1),
                               intrinsics, extrinsics, post_rots=post_rots, post_trans=post_trans)
        return self.lift_splat(depth, context, points), depth

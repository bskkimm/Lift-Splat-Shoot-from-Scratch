import torch
import torch.nn as nn
from .backbone import ImageBackbone
from .depth import DepthContextNet
from .lift_splat import LiftSplat


class LSS(nn.Module):
    def __init__(self, depth_bins=8, feature_channels=64, bev_bounds=((-8., 8., 1.), (-8., 8., 1.), (-2., 2., 1.))):
        super().__init__()
        self.backbone = ImageBackbone(channels=feature_channels)
        self.depth = DepthContextNet(feature_channels, depth_bins, feature_channels)
        self.lift_splat = LiftSplat(*bev_bounds)

    def forward(self, images, intrinsics, extrinsics, depths):
        features = self.backbone(images)
        depth, context = self.depth(features)
        points = __import__("lss.geometry", fromlist=["camera_to_ego"]).camera_to_ego(__import__("lss.geometry", fromlist=["make_frustum"]).make_frustum(features.shape[-2], features.shape[-1], depths, images.device)[None, None].expand(images.shape[0], images.shape[1], -1, -1, -1, -1), intrinsics, extrinsics)
        return self.lift_splat(depth, context, points), depth

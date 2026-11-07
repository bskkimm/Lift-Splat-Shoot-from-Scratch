"""Shared single-batch optimization and validation for BEV segmentation."""

import math
from dataclasses import dataclass

import torch

from lss.losses import binary_segmentation_loss
from lss.metrics import segmentation_intersection_union


@dataclass(frozen=True)
class SegmentationStepResult:
    """A mean batch loss and additive pixel counts measured before an update."""

    loss: float
    samples: int
    intersection: int
    union: int


@dataclass(frozen=True)
class SegmentationEpochResult:
    """Sample-weighted loss and dataset-level occupancy counts for an epoch."""

    loss: float
    samples: int
    intersection: int
    union: int
    iou: float | None


def _forward_batch(model, batch, device, pos_weight):
    images = batch["images"].to(device)
    intrinsics = batch["intrinsics"].to(device)
    extrinsics = batch["extrinsics"].to(device)
    targets = batch["bev_target"].to(device)
    post_rots = batch.get("post_rots")
    post_trans = batch.get("post_trans")
    if post_rots is not None:
        post_rots = post_rots.to(device)
    if post_trans is not None:
        post_trans = post_trans.to(device)
    logits = model(images, intrinsics, extrinsics, post_rots=post_rots, post_trans=post_trans)
    loss = binary_segmentation_loss(logits, targets, pos_weight=pos_weight)
    intersection, union = segmentation_intersection_union(logits.detach(), targets)
    result = SegmentationStepResult(float(loss.detach().item()), images.shape[0], intersection, union)
    return loss, result


def train_segmentation_step(model, batch, optimizer, device, pos_weight=1.0, grad_clip_norm=None):
    """Run one optimizer update and return pre-update loss and IoU counts."""
    if grad_clip_norm is not None and (not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0):
        raise ValueError("grad_clip_norm must be finite and positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, result = _forward_batch(model, batch, device, pos_weight)
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return result


def evaluate_segmentation_step(model, batch, device, pos_weight=1.0):
    """Evaluate without gradients and restore every module's training flag."""
    previous_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            _, result = _forward_batch(model, batch, device, pos_weight)
        return result
    finally:
        for module, previous_mode in previous_modes:
            module.training = previous_mode


def run_segmentation_epoch(model, loader, device, optimizer=None, pos_weight=1.0, grad_clip_norm=None):
    """Run an epoch, aggregating sample-weighted loss and pixel-level IoU."""
    if optimizer is None and grad_clip_norm is not None:
        raise ValueError("gradient clipping is only available during training")
    total_loss = 0.0
    samples = intersection = union = 0
    for batch in loader:
        if optimizer is None:
            result = evaluate_segmentation_step(model, batch, device, pos_weight=pos_weight)
        else:
            result = train_segmentation_step(model, batch, optimizer, device,
                                             pos_weight=pos_weight, grad_clip_norm=grad_clip_norm)
        total_loss += result.loss * result.samples
        samples += result.samples
        intersection += result.intersection
        union += result.union
    if samples == 0:
        raise ValueError("cannot compute an epoch over an empty loader")
    iou = intersection / union if union else None
    return SegmentationEpochResult(total_loss / samples, samples, intersection, union, iou)

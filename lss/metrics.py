"""Metrics for binary bird's-eye-view occupancy segmentation."""

import torch


@torch.no_grad()
def segmentation_intersection_union(logits, targets):
    """Return integer counts; sum these across batches before computing IoU."""
    if logits.shape != targets.shape:
        raise ValueError("segmentation logits and targets must have matching shapes")
    predicted = logits > 0
    target = targets.bool()
    return int((predicted & target).sum().item()), int((predicted | target).sum().item())


def segmentation_iou(intersection, union):
    """Compute IoU from aggregate counts; an empty union scores 1.0."""
    if intersection < 0 or union < 0 or intersection > union:
        raise ValueError("invalid segmentation intersection or union")
    return intersection / union if union else 1.0

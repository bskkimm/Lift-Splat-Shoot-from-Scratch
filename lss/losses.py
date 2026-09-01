import torch.nn.functional as F
import torch


def depth_loss(pred, target): return F.cross_entropy(pred, target)
def detection_loss(pred, target): return F.smooth_l1_loss(pred, target)


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    prob = logits.sigmoid(); ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()


def lss_loss(predictions, targets, depth_weight=1.0, detection_weight=1.0):
    return depth_weight * depth_loss(predictions["depth"], targets["depth"]) + detection_weight * detection_loss(predictions["boxes"], targets["boxes"])

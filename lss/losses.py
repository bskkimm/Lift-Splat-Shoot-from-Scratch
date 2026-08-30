import torch.nn.functional as F


def depth_loss(pred, target): return F.cross_entropy(pred, target)
def detection_loss(pred, target): return F.smooth_l1_loss(pred, target)

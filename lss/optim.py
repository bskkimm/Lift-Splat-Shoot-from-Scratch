import torch

__all__ = ["build_optimizer", "build_scheduler"]


def build_optimizer(model, lr=2e-4, weight_decay=0.01):
    if lr <= 0 or weight_decay < 0: raise ValueError("invalid optimizer hyperparameters")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

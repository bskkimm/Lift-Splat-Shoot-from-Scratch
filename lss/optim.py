import torch


def build_optimizer(model, lr=2e-4, weight_decay=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

import torch


def train_step(model, optimizer, batch, target):
    optimizer.zero_grad(); prediction = model(*batch); loss = target(prediction); loss.backward(); optimizer.step(); return float(loss.detach())


@torch.no_grad()
def evaluate(model, batch): return model(*batch)

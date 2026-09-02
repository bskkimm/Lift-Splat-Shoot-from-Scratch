import torch
from pathlib import Path


def train_step(model, optimizer, batch, target):
    optimizer.zero_grad(); prediction = model(*batch); loss = target(prediction); loss.backward(); optimizer.step(); return float(loss.detach())


@torch.no_grad()
def evaluate(model, batch): return model(*batch)


def save_checkpoint(model, optimizer, path, epoch):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def fit(model, loader, optimizer, loss_fn, epochs, checkpoint_dir=None, use_amp=False, scaler=None, start_epoch=0):
    scaler = scaler or torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []
    for epoch in range(start_epoch, epochs):
        total = 0.0
        for batch, target in loader: total += train_step(model, optimizer, batch, lambda output: loss_fn(output, target))
        history.append(total / max(1, len(loader)))
        if checkpoint_dir: save_checkpoint(model, optimizer, Path(checkpoint_dir) / f"epoch_{epoch + 1:04d}.pt", epoch + 1)
    return history

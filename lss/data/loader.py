from torch.utils.data import DataLoader
import random
import torch
from .collate import collate_camera_samples


def build_loader(dataset, batch_size=1, shuffle=False, num_workers=0, seed=None, pin_memory=False, persistent_workers=False, drop_last=False, timeout=0, **kwargs):
    def worker_init(worker_id):
        value = (seed or 0) + worker_id; random.seed(value); torch.manual_seed(value)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers and num_workers > 0, drop_last=drop_last, timeout=timeout, collate_fn=collate_camera_samples, worker_init_fn=worker_init if seed is not None else None, **kwargs)

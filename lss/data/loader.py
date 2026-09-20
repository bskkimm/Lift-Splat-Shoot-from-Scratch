from torch.utils.data import DataLoader
from .collate import collate_camera_samples


def build_loader(dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_camera_samples, **kwargs)

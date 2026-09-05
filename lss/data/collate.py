import torch


def collate_camera_samples(samples):
    return {
        "images": torch.stack([sample["images"] for sample in samples]),
        "intrinsics": torch.stack([sample["intrinsics"] for sample in samples]),
        "extrinsics": torch.stack([sample["extrinsics"] for sample in samples]),
        "boxes": [sample["boxes"] for sample in samples],
        "labels": [sample["labels"] for sample in samples],
    }

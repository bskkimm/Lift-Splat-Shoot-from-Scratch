import torch


def collate_camera_samples(samples):
    max_boxes = max((sample["boxes"].shape[0] for sample in samples), default=0)
    boxes = torch.zeros(len(samples), max_boxes, 9)
    labels = torch.full((len(samples), max_boxes), -1, dtype=torch.long)
    mask = torch.zeros(len(samples), max_boxes, dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = sample["boxes"].shape[0]; boxes[index, :count] = sample["boxes"]; labels[index, :count] = sample["labels"]; mask[index, :count] = True
    return {
        "images": torch.stack([sample["images"] for sample in samples]),
        "intrinsics": torch.stack([sample["intrinsics"] for sample in samples]),
        "extrinsics": torch.stack([sample["extrinsics"] for sample in samples]),
        "boxes": boxes, "labels": labels, "box_mask": mask,
    }

import torch


def collate_camera_samples(samples):
    if not samples: raise ValueError("cannot collate an empty sample list")
    if len({sample["images"].shape[0] for sample in samples}) != 1: raise ValueError("all samples must have the same camera count")
    max_boxes = max((sample["boxes"].shape[0] for sample in samples), default=0)
    boxes = torch.zeros(len(samples), max_boxes, 9)
    labels = torch.full((len(samples), max_boxes), -1, dtype=torch.long)
    mask = torch.zeros(len(samples), max_boxes, dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = sample["boxes"].shape[0]; boxes[index, :count] = sample["boxes"]; labels[index, :count] = sample["labels"]; mask[index, :count] = True
    result = {
        "images": torch.stack([sample["images"] for sample in samples]),
        "intrinsics": torch.stack([sample["intrinsics"] for sample in samples]),
        "extrinsics": torch.stack([sample["extrinsics"] for sample in samples]),
        "boxes": boxes, "labels": labels, "box_mask": mask,
    }
    has_target = ["bev_target" in sample for sample in samples]
    if any(has_target) and not all(has_target):
        raise ValueError("bev_target must be present in every sample or none")
    if all(has_target):
        result["bev_target"] = torch.stack([sample["bev_target"] for sample in samples])
    return result

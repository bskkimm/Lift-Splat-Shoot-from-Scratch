from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class NuScenesCameraDataset(Dataset):
    """Small JSON-indexed camera dataset; the index contains image paths and calibration."""
    def __init__(self, records, image_size=None):
        self.records, self.image_size = records, image_size

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = []
        for path in record["image_paths"]:
            image = Image.open(Path(path)).convert("RGB")
            if self.image_size: image = image.resize(self.image_size[::-1])
            images.append(torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255)
        return {"images": torch.stack(images), "intrinsics": torch.tensor(record["intrinsics"], dtype=torch.float32), "extrinsics": torch.tensor(record["extrinsics"], dtype=torch.float32), "boxes": torch.tensor(record.get("boxes", []), dtype=torch.float32), "labels": torch.tensor(record.get("labels", []), dtype=torch.long)}

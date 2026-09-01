from pathlib import Path
import json
from PIL import Image
import torch
from torch.utils.data import Dataset


class NuScenesCameraDataset(Dataset):
    """Small JSON-indexed camera dataset; the index contains image paths and calibration."""
    def __init__(self, records=None, image_size=None, dataroot=None, version="v1.0-trainval"):
        self.image_size = image_size
        if dataroot is not None:
            root = Path(dataroot).expanduser() / "v1.0-trainval"
            with open(root / "sample.json") as handle: samples = {x["token"]: x for x in json.load(handle)}
            with open(root / "sample_data.json") as handle: sample_data = {x["token"]: x for x in json.load(handle)}
            self.records = []
            for sample in samples.values():
                paths = []
                for token in sample["data"].values(): paths.append(str(Path(dataroot).expanduser() / sample_data[token]["filename"]))
                self.records.append({"token": sample["token"], "image_paths": paths, "intrinsics": [], "extrinsics": [], "boxes": [], "labels": []})
        else:
            self.records = records or []

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images = []
        for path in record["image_paths"]:
            image = Image.open(Path(path)).convert("RGB")
            if self.image_size: image = image.resize(self.image_size[::-1])
            images.append(torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255)
        return {"images": torch.stack(images), "intrinsics": torch.tensor(record["intrinsics"], dtype=torch.float32), "extrinsics": torch.tensor(record["extrinsics"], dtype=torch.float32), "boxes": torch.tensor(record.get("boxes", []), dtype=torch.float32), "labels": torch.tensor(record.get("labels", []), dtype=torch.long)}

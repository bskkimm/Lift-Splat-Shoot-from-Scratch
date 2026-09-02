from pathlib import Path
import json
from PIL import Image
import torch
from .categories import category_id
from lss.geometry import quaternion_to_matrix
from torch.utils.data import Dataset


class NuScenesCameraDataset(Dataset):
    """Small JSON-indexed camera dataset; the index contains image paths and calibration."""
    def __init__(self, records=None, image_size=None, dataroot=None, version="v1.0-trainval"):
        self.image_size = image_size
        if dataroot is not None:
            root = Path(dataroot).expanduser() / "v1.0-trainval"
            with open(root / "sample.json") as handle: samples = {x["token"]: x for x in json.load(handle)}
            with open(root / "sample_data.json") as handle: sample_data = {x["token"]: x for x in json.load(handle)}
            annotation_path = root / "sample_annotation.json"
            annotations = {}
            if annotation_path.exists():
                with open(annotation_path) as handle:
                    for item in json.load(handle): annotations.setdefault(item["sample_token"], []).append(item)
            calib_path = root / "calibrated_sensor.json"
            calibrations = {}
            if calib_path.exists():
                with open(calib_path) as handle: calibrations = {x["token"]: x for x in json.load(handle)}
            self.records = []
            for sample in samples.values():
                paths = []
                intrinsics, extrinsics = [], []
                for token in sample["data"].values():
                    item = sample_data[token]; paths.append(str(Path(dataroot).expanduser() / item["filename"]))
                    calib = calibrations.get(item.get("calibrated_sensor_token"), {})
                    intrinsics.append(calib.get("camera_intrinsic", [[1,0,0],[0,1,0],[0,0,1]]))
                    rotation = quaternion_to_matrix(calib.get("rotation", [1, 0, 0, 0])).tolist(); translation = calib.get("translation", [0, 0, 0])
                    extrinsics.append([rotation[0] + [translation[0]], rotation[1] + [translation[1]], rotation[2] + [translation[2]], [0,0,0,1]])
                anns = annotations.get(sample["token"], [])
                boxes = [a.get("translation", []) + a.get("size", []) + [a.get("rotation", [1,0,0,0])[0]] + list(a.get("velocity", [0, 0])[:2]) for a in anns]
                labels = [category_id(a.get("category_name", "")) for a in anns]
                self.records.append({"token": sample["token"], "image_paths": paths, "intrinsics": intrinsics, "extrinsics": extrinsics, "boxes": boxes, "labels": labels})
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

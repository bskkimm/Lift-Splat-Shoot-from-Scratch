from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from .categories import category_id, yaw_from_quaternion
from lss.geometry import quaternion_to_matrix
from .transforms import normalize_images
from .nuscenes_index import index_camera_keyframes, index_lidar_reference_poses
from .bev_targets import rasterize_polygons, vehicle_bottom_polygon
from .image_augmentation import augment_image
from .metadata_stream import load_sample_data
from torch.utils.data import Dataset


class NuScenesCameraDataset(Dataset):
    """Small JSON-indexed camera dataset; the index contains image paths and calibration."""
    CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT")

    @classmethod
    def has_camera_data(cls, record):
        return bool(record.get("image_paths"))
    def __init__(self, records=None, image_size=None, dataroot=None, version="v1.0-trainval", bev_bounds=None, split=None, augmentation=None, camera_count=None, training=False):
        if image_size is not None and (len(image_size) != 2 or min(image_size) <= 0): raise ValueError("image_size must be (height, width) with positive values")
        if image_size is not None and augmentation is not None:
            raise ValueError("augmentation cannot be combined with image_size")
        if split is not None and dataroot is None:
            raise ValueError("a named split requires a nuScenes dataroot")
        if split is not None and (version not in ("v1.0-mini", "v1.0-trainval") or split not in ("train", "val")):
            raise ValueError("supported version/split pairs are mini or trainval with train or val")
        self.image_size = image_size
        self.augmentation = augmentation
        self.camera_count = camera_count
        self.training = training
        self.bev_bounds = bev_bounds
        self.split = split
        if dataroot is not None:
            root = Path(dataroot).expanduser() / version
            if not root.is_dir(): raise FileNotFoundError(f"nuScenes version directory not found: {root}")
            with open(root / "sample.json") as handle: samples = {x["token"]: x for x in json.load(handle)}
            if split is not None:
                from nuscenes.utils.splits import create_splits_scenes

                with open(root / "scene.json") as handle:
                    scenes = {item["token"]: item["name"] for item in json.load(handle)}
                split_name = f"mini_{split}" if version == "v1.0-mini" else split
                selected_names = set(create_splits_scenes()[split_name])
                samples = {token: sample for token, sample in samples.items()
                           if scenes[sample["scene_token"]] in selected_names}
            raw_tokens = {token for token, sample in samples.items() if "data" not in sample}
            explicit_tokens = {item for sample in samples.values() if "data" in sample
                               for item in sample["data"].values()}
            sample_data = load_sample_data(root / "sample_data.json", raw_tokens, explicit_tokens)
            pose_path = root / "ego_pose.json"; poses = {}
            if pose_path.exists():
                with open(pose_path) as handle: poses = {x["token"]: x for x in json.load(handle)}
            annotation_path = root / "sample_annotation.json"
            annotations = {}
            if annotation_path.exists():
                with open(annotation_path) as handle:
                    annotation_records = json.load(handle)
                raw_annotations = [item for item in annotation_records
                                   if "category_name" not in item and "instance_token" in item]
                if raw_annotations:
                    with open(root / "instance.json") as handle:
                        instances = {item["token"]: item for item in json.load(handle)}
                    with open(root / "category.json") as handle:
                        categories = {item["token"]: item for item in json.load(handle)}
                    for item in raw_annotations:
                        instance = instances[item["instance_token"]]
                        item["category_name"] = categories[instance["category_token"]]["name"]
                for item in annotation_records:
                    annotations.setdefault(item["sample_token"], []).append(item)
            calib_path = root / "calibrated_sensor.json"
            calibrations = {}
            if calib_path.exists():
                with open(calib_path) as handle: calibrations = {x["token"]: x for x in json.load(handle)}
            camera_index = {}
            reference_poses = {}
            if any("data" not in sample for sample in samples.values()):
                with open(root / "sensor.json") as handle:
                    sensors = {item["token"]: item for item in json.load(handle)}
                raw_data = {token: item for token, item in sample_data.items()
                            if item.get("sample_token") in raw_tokens}
                camera_index = index_camera_keyframes(raw_data, calibrations, sensors)
                reference_poses = index_lidar_reference_poses(raw_data, calibrations, sensors)
            self.records = []
            for sample in samples.values():
                paths = []
                intrinsics, extrinsics = [], []
                reference_token = reference_poses.get(sample["token"])
                if "data" not in sample and reference_token is None:
                    raise ValueError(f"missing LIDAR_TOP keyframe for sample {sample['token']}")
                reference = poses[reference_token] if reference_token is not None else None
                if reference is not None:
                    ref_rotation = quaternion_to_matrix(reference["rotation"])
                    ref_translation = torch.tensor(reference["translation"], dtype=torch.float32)
                camera_names = self.CAMERA_ORDER
                data = sample.get("data", camera_index.get(sample["token"], {}))
                tokens = [data[name] for name in camera_names if name in data]
                tokens += [token for name, token in data.items() if name not in camera_names]
                for token in tokens:
                    item = sample_data[token]; paths.append(str(Path(dataroot).expanduser() / item["filename"]))
                    calib = calibrations.get(item.get("calibrated_sensor_token"), {})
                    intrinsics.append(calib.get("camera_intrinsic", [[1,0,0],[0,1,0],[0,0,1]]))
                    rotation = quaternion_to_matrix(calib.get("rotation", [1, 0, 0, 0])).tolist(); translation = calib.get("translation", [0, 0, 0])
                    pose = poses[item["ego_pose_token"]] if reference is not None else poses.get(item.get("ego_pose_token"), {})
                    pose_rotation = quaternion_to_matrix(pose.get("rotation", [1,0,0,0]))
                    pose_translation = torch.tensor(pose.get("translation", [0,0,0]), dtype=torch.float32)
                    global_rotation = pose_rotation @ torch.tensor(rotation, dtype=torch.float32)
                    global_translation = pose_rotation @ torch.tensor(translation, dtype=torch.float32) + pose_translation
                    if reference is not None:
                        global_rotation = ref_rotation.T @ global_rotation
                        global_translation = ref_rotation.T @ (global_translation - ref_translation)
                    rotation = global_rotation.tolist(); translation = global_translation.tolist()
                    extrinsics.append([rotation[0] + [translation[0]], rotation[1] + [translation[1]], rotation[2] + [translation[2]], [0,0,0,1]])
                anns = annotations.get(sample["token"], [])
                boxes = []
                for a in anns:
                    center = a.get("translation", [])
                    yaw = yaw_from_quaternion(a.get("rotation", [1,0,0,0]))
                    velocity = list(a.get("velocity", [0, 0]))
                    if reference is not None:
                        center = (ref_rotation.T @ (torch.tensor(center, dtype=torch.float32) - ref_translation)).tolist()
                        orientation = ref_rotation.T @ quaternion_to_matrix(a["rotation"])
                        yaw = torch.atan2(orientation[1, 0], orientation[0, 0]).item()
                        velocity_3d = (velocity + [0, 0, 0])[:3]
                        velocity = (ref_rotation.T @ torch.tensor(velocity_3d, dtype=torch.float32)).tolist()
                    boxes.append(center + a.get("size", []) + [yaw] + velocity[:2])
                labels = [category_id(a.get("category_name", "")) for a in anns]
                filtered = [(box, label) for box, label in zip(boxes, labels) if label >= 0]
                boxes, labels = [item[0] for item in filtered], [item[1] for item in filtered]
                record = {"token": sample["token"], "image_paths": paths, "intrinsics": intrinsics, "extrinsics": extrinsics, "boxes": boxes, "labels": labels}
                if bev_bounds is not None:
                    if reference is None:
                        raise ValueError(f"BEV targets require a reference ego pose for sample {sample['token']}")
                    target_rotation = quaternion_to_matrix(torch.tensor(reference["rotation"], dtype=torch.float64))
                    target_translation = torch.tensor(reference["translation"], dtype=torch.float64)
                    record["vehicle_polygons"] = [vehicle_bottom_polygon(a, target_rotation, target_translation)
                                                  for a in anns if a.get("category_name", "").split(".")[0] == "vehicle"]
                self.records.append(record)
        else:
            self.records = records or []

    def __len__(self): return len(self.records)

    @property
    def record_count(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self.bev_bounds is not None and "vehicle_polygons" not in record:
            raise ValueError("BEV targets require vehicle polygons in a reference ego frame")
        count = len(record["image_paths"])
        if len(record["intrinsics"]) != count or len(record["extrinsics"]) != count:
            raise ValueError("camera image and calibration counts must match")
        if self.camera_count is not None and (not isinstance(self.camera_count, int) or self.camera_count <= 0):
            raise ValueError("camera_count must be a positive integer")
        if self.training and self.camera_count is not None and self.camera_count > count:
            raise ValueError("camera_count cannot exceed available cameras")
        indices = (np.random.choice(count, self.camera_count, replace=False).tolist()
                   if self.training and self.camera_count is not None and self.camera_count < count
                   else list(range(count)))
        images = []
        intrinsics = torch.tensor([record["intrinsics"][i] for i in indices], dtype=torch.float32)
        augmentations = record.get("augmentations")
        if augmentations is not None and self.augmentation is not None:
            raise ValueError("record augmentations cannot be combined with an augmentation policy")
        if augmentations is not None and (self.image_size is not None or len(augmentations) != len(record["image_paths"])):
            raise ValueError("per-camera augmentations must match image count and cannot be combined with image_size")
        post_rots, post_trans = [], []
        for camera_index in indices:
            path = record["image_paths"][camera_index]
            if not Path(path).is_file(): raise FileNotFoundError(f"camera image not found: {path}")
            image = Image.open(Path(path)).convert("RGB")
            if self.augmentation is not None:
                image, rotation, translation = self.augmentation(image)
                post_rots.append(rotation)
                post_trans.append(translation)
            elif augmentations is not None:
                image, rotation, translation = augment_image(image, **augmentations[camera_index])
                post_rots.append(rotation)
                post_trans.append(translation)
            elif self.image_size:
                old_w, old_h = image.size; image = image.resize(self.image_size[::-1]); intrinsics[len(images), 0] *= self.image_size[1] / old_w; intrinsics[len(images), 1] *= self.image_size[0] / old_h
            images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous().float() / 255)
        boxes = torch.tensor(record.get("boxes", []), dtype=torch.float32).reshape(-1, 9)
        labels = torch.tensor(record.get("labels", []), dtype=torch.long).reshape(-1)
        result = {"images": normalize_images(torch.stack(images)), "intrinsics": intrinsics, "extrinsics": torch.tensor([record["extrinsics"][i] for i in indices], dtype=torch.float32), "boxes": boxes, "labels": labels}
        if augmentations is not None or self.augmentation is not None:
            result["post_rots"] = torch.stack(post_rots)
            result["post_trans"] = torch.stack(post_trans)
        if self.bev_bounds is not None:
            result["bev_target"] = rasterize_polygons(record["vehicle_polygons"], *self.bev_bounds)
        return result

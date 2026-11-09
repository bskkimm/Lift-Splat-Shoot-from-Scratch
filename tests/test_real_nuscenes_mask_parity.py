"""Opt-in comparison against nuScenes Box geometry; no model training."""

import json
import os
from pathlib import Path

import cv2
import ijson
import numpy as np
import pytest
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from lss.data.nuscenes_dataset import NuScenesCameraDataset


def stream_records(path):
    with path.open("rb") as handle:
        yield from ijson.items(handle, "item", use_float=True)


@pytest.mark.skipif(not os.environ.get("LSS_REAL_NUSCENES_ROOT"),
                    reason="set LSS_REAL_NUSCENES_ROOT for real-data target parity")
def test_first_validation_vehicle_target_matches_devkit_box_rasterization():
    root = Path(os.environ["LSS_REAL_NUSCENES_ROOT"]).expanduser()
    tables = root / "v1.0-trainval"
    dataset = NuScenesCameraDataset(dataroot=root, version="v1.0-trainval", split="val",
                                    image_size=(128, 352),
                                    bev_bounds=((-50, 50, 0.5), (-50, 50, 0.5)))
    sample_token = dataset.records[0]["token"]
    actual = dataset[0]["bev_target"][0].numpy()

    sensors = {row["token"]: row for row in json.loads((tables / "sensor.json").read_text())}
    calibrations = {row["token"]: row for row in json.loads((tables / "calibrated_sensor.json").read_text())}
    lidar_calibrations = {token for token, row in calibrations.items()
                          if sensors[row["sensor_token"]]["channel"] == "LIDAR_TOP"}
    lidar = next(row for row in stream_records(tables / "sample_data.json")
                 if row["sample_token"] == sample_token and row["is_key_frame"]
                 and row["calibrated_sensor_token"] in lidar_calibrations)
    pose = next(row for row in stream_records(tables / "ego_pose.json")
                if row["token"] == lidar["ego_pose_token"])
    instances = {row["token"]: row for row in json.loads((tables / "instance.json").read_text())}
    categories = {row["token"]: row["name"] for row in json.loads((tables / "category.json").read_text())}
    expected = np.zeros((200, 200), dtype=np.uint8)
    vehicle_count = 0
    for annotation in stream_records(tables / "sample_annotation.json"):
        if annotation["sample_token"] != sample_token:
            continue
        category = categories[instances[annotation["instance_token"]]["category_token"]]
        if not category.startswith("vehicle."):
            continue
        box = Box(annotation["translation"], annotation["size"], Quaternion(annotation["rotation"]))
        box.translate(-np.asarray(pose["translation"]))
        box.rotate(Quaternion(pose["rotation"]).inverse)
        xy = box.bottom_corners()[:2].T
        grid = np.round((xy - np.array([-50., -50.])) / 0.5).astype(np.int32)
        cv2.fillPoly(expected, [grid[:, [1, 0]]], 1)
        vehicle_count += 1
    assert vehicle_count > 0, "selected sample has no vehicles; parity would be vacuous"
    np.testing.assert_array_equal(actual, expected)

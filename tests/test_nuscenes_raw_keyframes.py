import json

import pytest
from PIL import Image

from lss.data.nuscenes_dataset import NuScenesCameraDataset


def write_table(root, name, records):
    (root / f"{name}.json").write_text(json.dumps(records))


def make_raw_dataset(tmp_path, duplicate=False):
    root = tmp_path / "v1.0-mini"
    root.mkdir()
    Image.new("RGB", (2, 2)).save(tmp_path / "camera.png")
    write_table(root, "sample", [{"token": "sample", "scene_token": "scene", "timestamp": 1}])
    records = [
        {"token": "camera", "sample_token": "sample", "is_key_frame": True,
         "calibrated_sensor_token": "cam-cal", "filename": "camera.png"},
        {"token": "sweep", "sample_token": "sample", "is_key_frame": False,
         "calibrated_sensor_token": "cam-cal", "filename": "sweep.png"},
        {"token": "radar", "sample_token": "sample", "is_key_frame": True,
         "calibrated_sensor_token": "radar-cal", "filename": "radar.pcd"},
    ]
    if duplicate:
        records.append({**records[0], "token": "duplicate"})
    write_table(root, "sample_data", records)
    write_table(root, "sensor", [
        {"token": "cam-sensor", "channel": "CAM_FRONT", "modality": "camera"},
        {"token": "radar-sensor", "channel": "RADAR_FRONT", "modality": "radar"},
    ])
    write_table(root, "calibrated_sensor", [
        {"token": "cam-cal", "sensor_token": "cam-sensor",
         "camera_intrinsic": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
         "rotation": [1, 0, 0, 0], "translation": [1, 0, 0]},
        {"token": "radar-cal", "sensor_token": "radar-sensor"},
    ])


def test_raw_nuscenes_uses_keyframe_camera_not_sweep_or_radar(tmp_path):
    make_raw_dataset(tmp_path)
    dataset = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")

    assert len(dataset) == 1
    assert dataset.records[0]["image_paths"] == [str(tmp_path / "camera.png")]
    item = dataset[0]
    assert item["images"].shape == (1, 3, 2, 2)
    assert item["intrinsics"][0, 0, 0] == 2


def test_duplicate_camera_keyframes_fail_explicitly(tmp_path):
    make_raw_dataset(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate.*CAM_FRONT"):
        NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")

import json
import math

import torch
import pytest

from lss.data.nuscenes_dataset import NuScenesCameraDataset
from tests.test_nuscenes_raw_keyframes import make_raw_dataset, write_table


def make_reference_fixture(tmp_path, omit_camera_pose=False):
    make_raw_dataset(tmp_path)
    root = tmp_path / "v1.0-mini"

    def read_table(name):
        return json.loads((root / f"{name}.json").read_text())

    sample_data = read_table("sample_data")
    if not omit_camera_pose:
        sample_data[0]["ego_pose_token"] = "camera-pose"
    else:
        del sample_data[0]["ego_pose_token"]
    next(item for item in sample_data if item["token"] == "lidar")["ego_pose_token"] = "reference-pose"
    write_table(root, "sample_data", sample_data)
    q = math.sqrt(0.5)
    write_table(root, "ego_pose", [
        {"token": "reference-pose", "translation": [10, 0, 0], "rotation": [q, 0, 0, q]},
        {"token": "camera-pose", "translation": [10, 2, 0], "rotation": [1, 0, 0, 0]},
    ])
    write_table(root, "sample_annotation", [
        {"token": "box", "sample_token": "sample", "category_name": "vehicle.car",
         "translation": [10, 3, 0], "size": [2, 4, 1], "rotation": [q, 0, 0, q],
         "velocity": [0, 2, 0]},
    ])


def test_raw_camera_and_box_use_lidar_reference_ego_frame(tmp_path):
    make_reference_fixture(tmp_path)

    item = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")[0]

    expected = torch.tensor([[0., 1., 0., 2.], [-1., 0., 0., -1.],
                             [0., 0., 1., 0.], [0., 0., 0., 1.]])
    torch.testing.assert_allclose(item["extrinsics"][0], expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_allclose(item["boxes"][0, :3], torch.tensor([3., 0., 0.]), rtol=1e-5, atol=1e-5)
    assert abs(item["boxes"][0, 6]) < 1e-5
    torch.testing.assert_allclose(item["boxes"][0, 7:9], torch.tensor([2., 0.]), rtol=1e-5, atol=1e-5)


def test_missing_camera_pose_fails_with_a_reference_frame(tmp_path):
    make_reference_fixture(tmp_path, omit_camera_pose=True)
    with pytest.raises(KeyError, match="ego_pose_token"):
        NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")


def test_raw_sample_without_lidar_reference_fails(tmp_path):
    make_raw_dataset(tmp_path)
    root = tmp_path / "v1.0-mini"
    records = json.loads((root / "sample_data.json").read_text())
    write_table(root, "sample_data", [item for item in records if item["token"] != "lidar"])
    with pytest.raises(ValueError, match="LIDAR_TOP"):
        NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")

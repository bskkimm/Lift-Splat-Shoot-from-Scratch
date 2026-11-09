"""Opt-in independent real-data camera-transform comparison; no training."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from pyquaternion import Quaternion

from lss.data.nuscenes_dataset import NuScenesCameraDataset
from tests.test_real_nuscenes_mask_parity import stream_records


@pytest.mark.skipif(not os.environ.get("LSS_REAL_NUSCENES_ROOT"),
                    reason="set LSS_REAL_NUSCENES_ROOT for camera transform parity")
def test_first_validation_camera_transforms_match_independent_pose_composition():
    root = Path(os.environ["LSS_REAL_NUSCENES_ROOT"]).expanduser()
    tables = root / "v1.0-trainval"
    dataset = NuScenesCameraDataset(dataroot=root, version="v1.0-trainval", split="val",
                                    image_size=(128, 352))
    sample_token = dataset.records[0]["token"]
    actual = dataset[0]
    sensors = {row["token"]: row for row in json.loads((tables / "sensor.json").read_text())}
    calibrations = {row["token"]: row for row in json.loads((tables / "calibrated_sensor.json").read_text())}
    selected = {}
    required = set(NuScenesCameraDataset.CAMERA_ORDER) | {"LIDAR_TOP"}
    for row in stream_records(tables / "sample_data.json"):
        if row["sample_token"] != sample_token or not row["is_key_frame"]:
            continue
        calibration = calibrations[row["calibrated_sensor_token"]]
        channel = sensors[calibration["sensor_token"]]["channel"]
        if channel in required:
            assert channel not in selected
            selected[channel] = row
    assert set(selected) == required
    pose_tokens = {row["ego_pose_token"] for row in selected.values()}
    poses = {row["token"]: row for row in stream_records(tables / "ego_pose.json")
             if row["token"] in pose_tokens}
    assert set(poses) == pose_tokens
    reference = poses[selected["LIDAR_TOP"]["ego_pose_token"]]
    ref_rotation = Quaternion(reference["rotation"]).rotation_matrix
    ref_translation = np.asarray(reference["translation"])
    for index, channel in enumerate(NuScenesCameraDataset.CAMERA_ORDER):
        row = selected[channel]
        pose = poses[row["ego_pose_token"]]
        calibration = calibrations[row["calibrated_sensor_token"]]
        pose_rotation = Quaternion(pose["rotation"]).rotation_matrix
        camera_rotation = Quaternion(calibration["rotation"]).rotation_matrix
        expected_rotation = ref_rotation.T @ pose_rotation @ camera_rotation
        expected_translation = ref_rotation.T @ (
            pose_rotation @ np.asarray(calibration["translation"])
            + np.asarray(pose["translation"]) - ref_translation
        )
        expected_intrinsics = np.asarray(calibration["camera_intrinsic"], dtype=np.float64).copy()
        expected_intrinsics[0] *= 352 / row["width"]
        expected_intrinsics[1] *= 128 / row["height"]
        np.testing.assert_allclose(actual["extrinsics"][index, :3, :3].numpy(), expected_rotation,
                                   rtol=0, atol=1e-5, err_msg=channel)
        np.testing.assert_allclose(actual["extrinsics"][index, :3, 3].numpy(), expected_translation,
                                   rtol=0, atol=1e-3, err_msg=channel)
        np.testing.assert_allclose(actual["intrinsics"][index].numpy(), expected_intrinsics,
                                   rtol=1e-6, atol=1e-5, err_msg=channel)

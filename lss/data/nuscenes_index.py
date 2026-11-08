"""Index key-frame cameras from the raw nuScenes metadata tables."""

from typing import Mapping, TypedDict


class SampleDataRecord(TypedDict):
    token: str
    sample_token: str
    is_key_frame: bool
    calibrated_sensor_token: str
    ego_pose_token: str


class CalibrationRecord(TypedDict):
    sensor_token: str


class SensorRecord(TypedDict):
    channel: str
    modality: str


def index_camera_keyframes(
    sample_data: Mapping[str, SampleDataRecord],
    calibrated_sensors: Mapping[str, CalibrationRecord],
    sensors: Mapping[str, SensorRecord],
) -> dict[str, dict[str, str]]:
    """Return ``sample_token -> channel -> sample_data_token`` for cameras.

    Tables are token-indexed dictionaries. Sweeps and non-camera sensors are
    intentionally excluded; multiple keyframes for one sample/channel are
    ambiguous and therefore rejected.
    """
    result: dict[str, dict[str, str]] = {}
    for item in sample_data.values():
        if not item["is_key_frame"]:
            continue
        calibration = calibrated_sensors[item["calibrated_sensor_token"]]
        sensor = sensors[calibration["sensor_token"]]
        if sensor["modality"] != "camera":
            continue
        channel = sensor["channel"]
        cameras = result.setdefault(item["sample_token"], {})
        if channel in cameras:
            raise ValueError(f"duplicate camera keyframe for {channel} in sample {item['sample_token']}")
        cameras[channel] = item["token"]
    return result


def index_lidar_reference_poses(
    sample_data: Mapping[str, SampleDataRecord],
    calibrated_sensors: Mapping[str, CalibrationRecord],
    sensors: Mapping[str, SensorRecord],
) -> dict[str, str]:
    """Return the ego-pose token for each sample's LIDAR_TOP keyframe."""
    result: dict[str, str] = {}
    for item in sample_data.values():
        if not item["is_key_frame"]:
            continue
        calibration = calibrated_sensors[item["calibrated_sensor_token"]]
        sensor = sensors[calibration["sensor_token"]]
        if sensor["channel"] != "LIDAR_TOP":
            continue
        sample_token = item["sample_token"]
        if sample_token in result:
            raise ValueError(f"duplicate LIDAR_TOP keyframe in sample {sample_token}")
        result[sample_token] = item["ego_pose_token"]
    return result

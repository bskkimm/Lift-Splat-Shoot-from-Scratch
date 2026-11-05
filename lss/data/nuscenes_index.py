"""Index key-frame cameras from the raw nuScenes metadata tables."""


def index_camera_keyframes(sample_data, calibrated_sensors, sensors):
    """Return ``sample_token -> channel -> sample_data_token`` for cameras.

    Tables are token-indexed dictionaries. Sweeps and non-camera sensors are
    intentionally excluded; multiple keyframes for one sample/channel are
    ambiguous and therefore rejected.
    """
    result = {}
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


def index_lidar_reference_poses(sample_data, calibrated_sensors, sensors):
    """Return the ego-pose token for each sample's LIDAR_TOP keyframe."""
    result = {}
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

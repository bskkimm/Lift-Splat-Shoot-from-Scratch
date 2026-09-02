import json
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_dataset_reads_camera_intrinsics(tmp_path):
    root = tmp_path / "v1.0-trainval"; root.mkdir()
    (root / "sample.json").write_text(json.dumps([{"token":"s","data":{"cam":"d"}}]))
    (root / "sample_data.json").write_text(json.dumps([{"token":"d","filename":"x.jpg","calibrated_sensor_token":"c"}]))
    (root / "calibrated_sensor.json").write_text(json.dumps([{"token":"c","camera_intrinsic":[[2,0,1],[0,2,1],[0,0,1]]}]))
    assert NuScenesCameraDataset(dataroot=tmp_path).records[0]["intrinsics"][0][0][0] == 2

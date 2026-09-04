import json
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_dataset_composes_ego_pose_translation(tmp_path):
    root=tmp_path/"v1.0-trainval"; root.mkdir()
    (root/"sample.json").write_text(json.dumps([{"token":"s","data":{"c":"d"}}])); (root/"sample_data.json").write_text(json.dumps([{"token":"d","filename":"x","calibrated_sensor_token":"c","ego_pose_token":"p"}]))
    (root/"calibrated_sensor.json").write_text(json.dumps([{"token":"c","translation":[0,0,0],"rotation":[1,0,0,0]}])); (root/"ego_pose.json").write_text(json.dumps([{"token":"p","translation":[3,0,0],"rotation":[1,0,0,0]}]))
    assert NuScenesCameraDataset(dataroot=tmp_path).records[0]["extrinsics"][0][0][3] == 3

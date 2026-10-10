import json
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_samples_without_cameras_are_detectable(tmp_path):
    root=tmp_path/"v1.0-trainval"; root.mkdir(); (root/"sample.json").write_text(json.dumps([{"token":"s","data":{}}])); (root/"sample_data.json").write_text("[]"); assert not NuScenesCameraDataset.has_camera_data(NuScenesCameraDataset(dataroot=tmp_path).records[0])

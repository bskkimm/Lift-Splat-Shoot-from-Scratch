import json
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_dataset_honors_version_argument(tmp_path):
    root=tmp_path/"v1.0-mini"; root.mkdir(); (root/"sample.json").write_text(json.dumps([])); (root/"sample_data.json").write_text("[]")
    assert len(NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")) == 0

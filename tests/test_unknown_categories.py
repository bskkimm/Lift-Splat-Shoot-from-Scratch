import json
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_unknown_categories_are_filtered(tmp_path):
    root=tmp_path/"v1.0-trainval"; root.mkdir(); (root/"sample.json").write_text(json.dumps([{"token":"s","data":{}}])); (root/"sample_data.json").write_text("[]"); (root/"sample_annotation.json").write_text(json.dumps([{"sample_token":"s","translation":[0,0,0],"size":[1,1,1],"category_name":"unknown"}]))
    assert NuScenesCameraDataset(dataroot=tmp_path).records[0]["labels"] == []

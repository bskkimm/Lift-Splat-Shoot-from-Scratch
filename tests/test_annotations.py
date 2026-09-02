import json
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_annotation_records_include_velocity_and_orientation(tmp_path):
    root=tmp_path/"v1.0-trainval"; root.mkdir()
    (root/"sample.json").write_text(json.dumps([{"token":"s","data":{}}]))
    (root/"sample_data.json").write_text("[]")
    (root/"sample_annotation.json").write_text(json.dumps([{"sample_token":"s","translation":[1,2,3],"size":[4,5,6],"rotation":[.7,0,0,.7],"velocity":[2,3,0],"category_name":"vehicle.car"}]))
    item=NuScenesCameraDataset(dataroot=tmp_path).records[0]; assert item["boxes"][0][-2:] == [2,3]

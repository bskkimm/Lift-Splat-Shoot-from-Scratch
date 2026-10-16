import json
from lss.evaluation import export_predictions
def test_export_adds_dataset_version(tmp_path):
    p=tmp_path/"r.json"; export_predictions({},p); assert json.loads(p.read_text())["meta"]["version"] == "v1.0-trainval"

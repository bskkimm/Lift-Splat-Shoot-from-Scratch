import json
from lss.evaluation import export_predictions
def test_export_adds_nuscenes_result_schema(tmp_path):
    path=tmp_path/"r.json"; export_predictions({},path); assert json.loads(path.read_text()) == {"meta":{},"results":[]}

import json
from lss.evaluation import export_predictions
def test_prediction_export_writes_json(tmp_path):
    path = tmp_path / "results.json"; export_predictions({"results": []}, path); assert json.loads(path.read_text()) == {"results": []}

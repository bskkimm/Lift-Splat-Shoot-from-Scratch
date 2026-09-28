import pytest
from lss.evaluation import export_predictions
def test_json_export_accepts_nested_json_path(tmp_path):
    path=tmp_path/"x"/"out.json"; export_predictions({"ok":True},path); assert path.exists()

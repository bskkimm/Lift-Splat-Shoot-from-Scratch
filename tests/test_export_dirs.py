from lss.evaluation import export_predictions
def test_export_predictions_creates_parent_dirs(tmp_path):
    path=tmp_path/"nested"/"results.json"; export_predictions({},path); assert path.exists()

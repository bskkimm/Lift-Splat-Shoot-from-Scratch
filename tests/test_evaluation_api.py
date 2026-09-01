from lss.evaluation import evaluate_nuscenes
def test_official_evaluation_is_callable():
    assert callable(evaluate_nuscenes)

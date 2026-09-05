from lss.models import BEVHead, LSS


def test_model_exports_are_available():
    assert BEVHead is not None and LSS is not None

import pytest
from lss.models.factory import build_model
def test_factory_rejects_invalid_depth_bins():
    with pytest.raises(ValueError): build_model(depth_bins=0)

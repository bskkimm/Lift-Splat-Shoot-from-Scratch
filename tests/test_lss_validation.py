import pytest
from lss.models.lss import LSS
def test_lss_rejects_invalid_feature_channels():
    with pytest.raises(ValueError): LSS(feature_channels=0)

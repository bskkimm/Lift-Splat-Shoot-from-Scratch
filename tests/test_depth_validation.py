import pytest
from lss.models.depth import DepthContextNet
def test_depth_network_rejects_invalid_bins():
    with pytest.raises(ValueError): DepthContextNet(depth_bins=0)

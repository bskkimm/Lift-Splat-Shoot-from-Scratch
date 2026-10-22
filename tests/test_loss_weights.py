import pytest
from lss.losses import lss_loss
def test_loss_rejects_negative_weights():
    with pytest.raises(ValueError): lss_loss({}, {}, depth_weight=-1)

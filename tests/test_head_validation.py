import pytest
from lss.models.head import BEVHead
def test_head_rejects_invalid_classes():
    with pytest.raises(ValueError): BEVHead(classes=0)

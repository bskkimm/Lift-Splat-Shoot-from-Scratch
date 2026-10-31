import pytest
from lss.models.backbone import ImageBackbone
def test_backbone_rejects_invalid_channels():
    with pytest.raises(ValueError): ImageBackbone(channels=0)

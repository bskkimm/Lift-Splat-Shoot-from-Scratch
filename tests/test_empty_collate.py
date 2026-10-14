import pytest
from lss.data.collate import collate_camera_samples
def test_empty_collate_is_explicit():
    with pytest.raises(ValueError): collate_camera_samples([])

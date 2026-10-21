import pytest
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_image_size_is_validated():
    with pytest.raises(ValueError): NuScenesCameraDataset([], image_size=(0,2))

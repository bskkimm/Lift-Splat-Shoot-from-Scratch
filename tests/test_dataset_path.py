import pytest
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_missing_dataset_path_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="version directory"): NuScenesCameraDataset(dataroot=tmp_path)

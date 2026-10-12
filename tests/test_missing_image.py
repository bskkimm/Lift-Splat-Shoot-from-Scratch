import pytest
from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_missing_camera_image_is_explicit():
    record={"image_paths":["missing.png"],"intrinsics":[[[1,0,0],[0,1,0],[0,0,1]]],"extrinsics":[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]}
    with pytest.raises(FileNotFoundError): NuScenesCameraDataset([record])[0]

from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_camera_data_validation(): assert not NuScenesCameraDataset.has_camera_data({"image_paths": []})

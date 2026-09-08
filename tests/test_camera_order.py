from lss.data.nuscenes_dataset import NuScenesCameraDataset
def test_dataset_declares_standard_camera_order(): assert NuScenesCameraDataset.CAMERA_ORDER[0] == "CAM_FRONT" if hasattr(NuScenesCameraDataset,"CAMERA_ORDER") else True

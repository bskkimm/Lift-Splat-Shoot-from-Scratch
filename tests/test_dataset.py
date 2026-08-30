from lss.data.nuscenes_dataset import NuScenesCameraDataset


def test_dataset_returns_camera_tensor_contract(tmp_path):
    from PIL import Image
    image = tmp_path / "cam.png"; Image.new("RGB", (4, 3)).save(image)
    record = {"image_paths": [str(image)], "intrinsics": [[[1,0,0],[0,1,0],[0,0,1]]], "extrinsics": [[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]}
    item = NuScenesCameraDataset([record])[0]
    assert item["images"].shape == (1, 3, 3, 4)

from lss.data.nuscenes_dataset import NuScenesCameraDataset


def test_dataset_returns_camera_tensor_contract(tmp_path):
    from PIL import Image
    image = tmp_path / "cam.png"; Image.new("RGB", (4, 3)).save(image)
    record = {"image_paths": [str(image)], "intrinsics": [[[1,0,0],[0,1,0],[0,0,1]]], "extrinsics": [[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]}
    item = NuScenesCameraDataset([record])[0]
    assert item["images"].shape == (1, 3, 3, 4)


def test_dataset_reads_nuscenes_sample_metadata(tmp_path):
    import json
    root = tmp_path / "v1.0-trainval"; root.mkdir(); (tmp_path / "cam.png").write_bytes(b"not-used")
    (root / "sample.json").write_text(json.dumps([{"token": "s", "data": {"CAM_FRONT": "d"}}]))
    (root / "sample_data.json").write_text(json.dumps([{"token": "d", "filename": "cam.png"}]))
    dataset = NuScenesCameraDataset(dataroot=tmp_path)
    assert dataset.records[0]["token"] == "s"

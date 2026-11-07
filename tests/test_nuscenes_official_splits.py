import json

import pytest

from lss.data.nuscenes_dataset import NuScenesCameraDataset
from tests.test_nuscenes_raw_keyframes import make_raw_dataset, write_table


def make_two_scene_dataset(tmp_path):
    make_raw_dataset(tmp_path)
    root = tmp_path / "v1.0-mini"
    write_table(root, "scene", [
        {"token": "train-scene", "name": "scene-0061"},
        {"token": "val-scene", "name": "scene-0103"},
    ])
    write_table(root, "sample", [
        {"token": "sample", "scene_token": "train-scene", "timestamp": 1},
        {"token": "val-sample", "scene_token": "val-scene", "timestamp": 2},
    ])
    sample_data = json.loads((root / "sample_data.json").read_text())
    camera = next(item for item in sample_data if item["token"] == "camera")
    lidar = next(item for item in sample_data if item["token"] == "lidar")
    sample_data.extend([
        {**camera, "token": "val-camera", "sample_token": "val-sample"},
        {**lidar, "token": "val-lidar", "sample_token": "val-sample"},
    ])
    write_table(root, "sample_data", sample_data)


def test_official_mini_train_and_val_scenes_are_disjoint(tmp_path):
    make_two_scene_dataset(tmp_path)
    train = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini", split="train")
    val = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini", split="val")
    assert [record["token"] for record in train.records] == ["sample"]
    assert [record["token"] for record in val.records] == ["val-sample"]


def test_no_split_is_explicitly_unfiltered(tmp_path):
    make_two_scene_dataset(tmp_path)
    dataset = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")
    assert {record["token"] for record in dataset.records} == {"sample", "val-sample"}


def test_official_trainval_split_uses_full_scene_lists(tmp_path):
    make_two_scene_dataset(tmp_path)
    root = tmp_path / "v1.0-mini"
    write_table(root, "scene", [
        {"token": "train-scene", "name": "scene-0001"},
        {"token": "val-scene", "name": "scene-0003"},
    ])
    root.rename(tmp_path / "v1.0-trainval")
    train = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-trainval", split="train")
    val = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-trainval", split="val")
    assert [record["token"] for record in train.records] == ["sample"]
    assert [record["token"] for record in val.records] == ["val-sample"]


@pytest.mark.parametrize("version,split", [
    ("v1.0-mini", "test"), ("v1.0-trainval", "mini_train"), ("v1.0-test", "train"),
])
def test_invalid_version_or_split_fails_explicitly(tmp_path, version, split):
    with pytest.raises(ValueError, match="split|version"):
        NuScenesCameraDataset(dataroot=tmp_path, version=version, split=split)


def test_named_split_rejects_preconstructed_records():
    with pytest.raises(ValueError, match="dataroot"):
        NuScenesCameraDataset(records=[], split="train")


def test_unknown_scene_token_is_not_silently_dropped(tmp_path):
    make_two_scene_dataset(tmp_path)
    root = tmp_path / "v1.0-mini"
    samples = json.loads((root / "sample.json").read_text())
    samples[0]["scene_token"] = "missing-scene"
    write_table(root, "sample", samples)
    with pytest.raises(KeyError, match="missing-scene"):
        NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini", split="train")

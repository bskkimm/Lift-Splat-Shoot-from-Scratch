import pytest

from lss.data.nuscenes_dataset import NuScenesCameraDataset
from tests.test_nuscenes_raw_keyframes import make_raw_dataset, write_table


def add_raw_annotations(tmp_path, missing_instance=False):
    root = tmp_path / "v1.0-mini"
    annotations = [
        {"token": "a", "sample_token": "sample", "instance_token": "car-instance",
         "translation": [1, 2, 3], "size": [2, 4, 1], "rotation": [1, 0, 0, 0]},
        {"token": "b", "sample_token": "sample", "instance_token": "animal-instance",
         "translation": [5, 6, 7], "size": [1, 1, 1], "rotation": [1, 0, 0, 0]},
        {"token": "c", "sample_token": "sample", "instance_token": "bus-instance",
         "translation": [8, 9, 0], "size": [2, 8, 3], "rotation": [1, 0, 0, 0]},
    ]
    write_table(root, "sample_annotation", annotations)
    instances = [
        {"token": "car-instance", "category_token": "car-category"},
        {"token": "animal-instance", "category_token": "animal-category"},
        {"token": "bus-instance", "category_token": "bus-category"},
    ]
    write_table(root, "instance", instances[1:] if missing_instance else instances)
    write_table(root, "category", [
        {"token": "car-category", "name": "vehicle.car"},
        {"token": "animal-category", "name": "animal"},
        {"token": "bus-category", "name": "vehicle.bus.rigid"},
    ])


def test_raw_annotation_category_is_resolved_through_instance(tmp_path):
    make_raw_dataset(tmp_path)
    add_raw_annotations(tmp_path)
    dataset = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")
    assert dataset.records[0]["labels"] == [0, 2]
    assert dataset.records[0]["boxes"][0][:6] == [1, 2, 3, 2, 4, 1]


def test_missing_raw_instance_is_not_silently_ignored(tmp_path):
    make_raw_dataset(tmp_path)
    add_raw_annotations(tmp_path, missing_instance=True)
    with pytest.raises(KeyError, match="car-instance"):
        NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")

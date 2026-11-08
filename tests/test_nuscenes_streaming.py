import json

from lss.data.nuscenes_dataset import NuScenesCameraDataset
from tests.test_nuscenes_raw_keyframes import make_raw_dataset, write_table


def test_streamed_sample_data_keeps_selected_keyframes_and_explicit_tokens(tmp_path):
    from lss.data.metadata_stream import load_sample_data

    path = tmp_path / "sample_data.json"
    records = [
        {"token": "camera", "sample_token": "selected", "is_key_frame": True},
        {"token": "sweep", "sample_token": "selected", "is_key_frame": False},
        {"token": "other", "sample_token": "other", "is_key_frame": True},
        {"token": "legacy"},
    ]
    path.write_text(json.dumps(records))
    result = load_sample_data(path, {"selected"}, {"legacy"})
    assert set(result) == {"camera", "legacy"}


def test_irrelevant_raw_rows_do_not_require_sensor_relations(tmp_path):
    make_raw_dataset(tmp_path)
    root = tmp_path / "v1.0-mini"
    records = json.loads((root / "sample_data.json").read_text())
    records.extend([
        {"token": "other", "sample_token": "other", "is_key_frame": True,
         "calibrated_sensor_token": "missing"},
        {"token": "sweep-broken", "sample_token": "sample", "is_key_frame": False,
         "calibrated_sensor_token": "missing"},
    ])
    write_table(root, "sample_data", records)
    dataset = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini")
    assert len(dataset) == 1
    assert dataset[0]["images"].shape[0] == 1

import json
import math

import pytest
import torch

from lss.data.nuscenes_dataset import NuScenesCameraDataset
from lss.data.collate import collate_camera_samples
from lss.geometry import quaternion_to_matrix
from tests.test_nuscenes_raw_keyframes import write_table
from tests.test_nuscenes_reference_frames import make_reference_fixture


def test_vehicle_bottom_corners_transform_into_reference_ego_frame():
    from lss.data.bev_targets import vehicle_bottom_polygon

    q = math.sqrt(0.5)
    annotation = {"translation": [10, 3, 0], "size": [2, 4, 1], "rotation": [q, 0, 0, q]}
    actual = vehicle_bottom_polygon(annotation, quaternion_to_matrix([q, 0, 0, q]), torch.tensor([10., 0., 0.]))
    expected = torch.tensor([[5., -1.], [5., 1.], [1., 1.], [1., -1.]])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_rasterization_uses_xy_grid_order_and_inclusive_polygon_edges():
    from lss.data.bev_targets import rasterize_polygons

    polygon = torch.tensor([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    target = rasterize_polygons([polygon], (-1, 3, 1), (-1, 5, 1))
    expected = torch.zeros(1, 4, 6)
    expected[0, 1:3, 1:3] = 1
    torch.testing.assert_close(target, expected)


def test_empty_polygons_produce_empty_bev_mask():
    from lss.data.bev_targets import rasterize_polygons

    target = rasterize_polygons([], (0, 4, 1), (0, 6, 1))
    assert target.shape == (1, 4, 6)
    assert target.sum() == 0


def test_rasterization_clips_and_unions_overlapping_polygons():
    from lss.data.bev_targets import rasterize_polygons

    first = torch.tensor([[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]])
    second = torch.tensor([[1., 1.], [1., 2.], [2., 2.], [2., 1.]])
    target = rasterize_polygons([first, second], (0, 4, 1), (0, 4, 1))
    assert target.sum().item() == 7


def test_bottom_corners_preserve_double_precision():
    from lss.data.bev_targets import vehicle_bottom_polygon

    q = math.sqrt(0.5)
    annotation = {"translation": [10, 3, 0], "size": [2, 4, 1], "rotation": [q, 0, 0, q]}
    rotation = quaternion_to_matrix(torch.tensor([q, 0, 0, q], dtype=torch.float64))
    actual = vehicle_bottom_polygon(annotation, rotation, torch.tensor([10., 0., 0.], dtype=torch.float64))
    expected = torch.tensor([[5., -1.], [5., 1.], [1., 1.], [1., -1.]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_dataset_vehicle_mask_uses_all_vehicle_categories(tmp_path):
    make_reference_fixture(tmp_path)
    root = tmp_path / "v1.0-mini"
    annotations = json.loads((root / "sample_annotation.json").read_text())
    q = math.sqrt(0.5)
    annotations.append({"token": "emergency", "sample_token": "sample",
                        "category_name": "vehicle.emergency.police", "translation": [10, 7, 0],
                        "size": [2, 2, 1], "rotation": [q, 0, 0, q]})
    write_table(root, "sample_annotation", annotations)

    dataset = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini",
                                    bev_bounds=((0, 8, 1), (-4, 4, 1)))
    item = dataset[0]

    assert item["labels"].tolist() == [0]  # Emergency vehicles are not a detection class.
    assert item["bev_target"].shape == (1, 8, 8)
    assert item["bev_target"].sum().item() == 21
    batch = collate_camera_samples([item, item])
    assert batch["bev_target"].shape == (2, 1, 8, 8)
    torch.testing.assert_close(batch["bev_target"][0], item["bev_target"])


def test_collate_rejects_mixed_presence_of_bev_targets(tmp_path):
    make_reference_fixture(tmp_path)
    dataset = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini",
                                    bev_bounds=((0, 8, 1), (-4, 4, 1)))
    item = dataset[0]
    without_target = {key: value for key, value in item.items() if key != "bev_target"}
    with pytest.raises(ValueError, match="bev_target"):
        collate_camera_samples([item, without_target])


def test_dataset_bev_target_requires_reference_pose():
    record = {"image_paths": [], "intrinsics": [], "extrinsics": [], "boxes": [], "labels": []}
    dataset = NuScenesCameraDataset([record], bev_bounds=((0, 8, 1), (0, 8, 1)))
    with pytest.raises(ValueError, match="reference"):
        dataset[0]

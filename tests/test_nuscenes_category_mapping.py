import pytest

from lss.data.categories import CATEGORY_TO_ID, category_id


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("movable_object.barrier", "barrier"),
        ("vehicle.bicycle", "bicycle"),
        ("vehicle.bus.bendy", "bus"),
        ("vehicle.bus.rigid", "bus"),
        ("vehicle.car", "car"),
        ("vehicle.construction", "construction_vehicle"),
        ("vehicle.motorcycle", "motorcycle"),
        ("human.pedestrian.adult", "pedestrian"),
        ("human.pedestrian.child", "pedestrian"),
        ("human.pedestrian.construction_worker", "pedestrian"),
        ("human.pedestrian.police_officer", "pedestrian"),
        ("movable_object.trafficcone", "traffic_cone"),
        ("vehicle.trailer", "trailer"),
        ("vehicle.truck", "truck"),
    ],
)
def test_raw_nuscenes_detection_category_mapping(source, target):
    assert category_id(source) == CATEGORY_TO_ID[target]


@pytest.mark.parametrize("name", [
    "human.pedestrian.stroller", "human.pedestrian.wheelchair",
    "human.pedestrian.personal_mobility", "other.car",
])
def test_categories_outside_detection_protocol_are_excluded(name):
    assert category_id(name) == -1

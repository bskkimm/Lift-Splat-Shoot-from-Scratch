CATEGORIES = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier")
CATEGORY_TO_ID = {name: index for index, name in enumerate(CATEGORIES)}

NUSCENES_DETECTION_CATEGORIES = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def category_id(name):
    """Map an exact raw nuScenes or detection class name to its class ID."""
    return CATEGORY_TO_ID.get(NUSCENES_DETECTION_CATEGORIES.get(name, name), -1)

def yaw_from_quaternion(q):
    import math
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

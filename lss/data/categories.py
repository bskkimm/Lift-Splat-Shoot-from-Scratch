CATEGORIES = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier")
CATEGORY_TO_ID = {name: index for index, name in enumerate(CATEGORIES)}


def category_id(name):
    name = name.rsplit(".", 1)[-1]
    return CATEGORY_TO_ID.get(name, -1)

def yaw_from_quaternion(q):
    import math
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

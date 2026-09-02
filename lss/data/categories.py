CATEGORIES = ("car", "truck", "bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier")
CATEGORY_TO_ID = {name: index for index, name in enumerate(CATEGORIES)}


def category_id(name):
    name = name.rsplit(".", 1)[-1]
    return CATEGORY_TO_ID.get(name, -1)

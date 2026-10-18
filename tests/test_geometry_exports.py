from lss.geometry import camera_to_ego, make_frustum, quaternion_to_matrix
def test_geometry_functions_are_importable(): assert all(callable(f) for f in (camera_to_ego,make_frustum,quaternion_to_matrix))

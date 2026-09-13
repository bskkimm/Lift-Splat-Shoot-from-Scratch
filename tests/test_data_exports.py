from lss.data import CATEGORIES, collate_camera_samples
def test_data_helpers_are_public(): assert len(CATEGORIES) == 10 and callable(collate_camera_samples)

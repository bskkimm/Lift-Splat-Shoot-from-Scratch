from lss.data.categories import category_id
def test_category_id_strips_nuscenes_namespace(): assert category_id("vehicle.car") == 0

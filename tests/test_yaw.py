from lss.data.categories import yaw_from_quaternion
def test_yaw_from_quaternion(): assert abs(yaw_from_quaternion([2**.5/2,0,0,2**.5/2]) - 1.5708) < .001

import torch
from lss.geometry import quaternion_to_matrix
def test_identity_quaternion_is_identity_matrix(): assert torch.equal(quaternion_to_matrix([1,0,0,0]), torch.eye(3))

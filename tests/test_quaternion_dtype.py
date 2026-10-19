import torch
from lss.geometry import quaternion_to_matrix
def test_quaternion_preserves_float_dtype(): assert quaternion_to_matrix(torch.tensor([1.,0,0,0],dtype=torch.float64)).dtype == torch.float64

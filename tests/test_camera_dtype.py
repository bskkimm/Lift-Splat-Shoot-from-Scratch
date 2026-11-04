import pytest, torch
from lss.data import CameraBatch
def test_camera_batch_rejects_integer_images():
    with pytest.raises(TypeError): CameraBatch(torch.zeros(1,1,3,1,1,dtype=torch.long),torch.eye(3)[None,None],torch.eye(4)[None,None])

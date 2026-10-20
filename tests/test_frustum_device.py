import torch
from lss.geometry import make_frustum
def test_frustum_uses_requested_device(): assert make_frustum(1,1,[1],"cpu").device.type == "cpu"

import pytest, torch
from lss.data import CameraBatch, Sample
def test_sample_rejects_mismatched_targets():
    camera=CameraBatch(torch.zeros(1,1,3,1,1),torch.eye(3)[None,None],torch.eye(4)[None,None])
    with pytest.raises(ValueError): Sample(camera,torch.zeros(2,9),torch.zeros(1,dtype=torch.long))

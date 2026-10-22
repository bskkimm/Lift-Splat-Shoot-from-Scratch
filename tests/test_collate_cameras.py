import pytest, torch
from lss.data.collate import collate_camera_samples
def test_collate_rejects_mismatched_camera_counts():
    s=lambda n:{"images":torch.zeros(n,3,1,1),"intrinsics":torch.zeros(n,3,3),"extrinsics":torch.zeros(n,4,4),"boxes":torch.zeros(0,9),"labels":torch.zeros(0,dtype=torch.long)}
    with pytest.raises(ValueError): collate_camera_samples([s(1),s(2)])

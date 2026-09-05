import torch
from lss.data.collate import collate_camera_samples
def test_collate_stacks_camera_tensors():
    sample={"images":torch.zeros(1,3,2,2),"intrinsics":torch.eye(3)[None],"extrinsics":torch.eye(4)[None],"boxes":torch.zeros(0,9),"labels":torch.zeros(0,dtype=torch.long)}
    assert collate_camera_samples([sample,sample])["images"].shape[0] == 2

import torch
from lss.data.collate import collate_camera_samples
def test_collate_pads_variable_box_counts():
    s={"images":torch.zeros(1,3,2,2),"intrinsics":torch.eye(3)[None],"extrinsics":torch.eye(4)[None],"boxes":torch.ones(2,9),"labels":torch.ones(2,dtype=torch.long)}; out=collate_camera_samples([s,{**s,"boxes":s["boxes"][:1],"labels":s["labels"][:1]}]); assert out["boxes"].shape==(2,2,9) and out["box_mask"].tolist()==[[True,True],[True,False]]

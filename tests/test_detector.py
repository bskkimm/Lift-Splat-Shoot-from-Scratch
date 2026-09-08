import torch
from lss.models.detector import LSSDetector
def test_detector_returns_detection_and_depth_outputs():
    model=LSSDetector(classes=2,depth_bins=2,bev_bounds=((-2,2,1),(-2,2,1),(-1,1,1))); out=model(torch.randn(1,1,3,8,8),torch.eye(3)[None,None],torch.eye(4)[None,None],[1.,2.]); assert out["logits"].shape[1]==2 and "depth" in out

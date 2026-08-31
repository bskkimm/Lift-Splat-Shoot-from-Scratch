import torch
from lss.models.lss import LSS
from lss.models.head import BEVHead


def test_synthetic_lss_detection_pipeline_is_differentiable():
    model = LSS(depth_bins=2, bev_bounds=((-2., 2., 1.), (-2., 2., 1.), (-1., 1., 1.)))
    bev, _ = model(torch.randn(1, 1, 3, 8, 8), torch.eye(3)[None, None], torch.eye(4)[None, None], [1., 2.])
    outputs = BEVHead()(bev)
    loss = outputs["logits"].mean() + outputs["boxes"].mean(); loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

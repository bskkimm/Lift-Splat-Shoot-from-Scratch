import pytest
import torch


def test_bev_encoder_preserves_grid_and_backpropagates():
    from lss.models.bev_encoder import BEVEncoder

    torch.manual_seed(0)
    model = BEVEncoder(in_channels=8, out_channels=2)
    bev = torch.randn(2, 8, 32, 40, requires_grad=True)

    logits = model(bev)

    assert logits.shape == (2, 2, 32, 40)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert bev.grad is not None and torch.isfinite(bev.grad).all()
    assert torch.count_nonzero(bev.grad) > 0


def test_bev_encoder_rejects_unaligned_grids():
    from lss.models.bev_encoder import BEVEncoder

    model = BEVEncoder(in_channels=8)
    with pytest.raises(ValueError, match="divisible by 8"):
        model(torch.randn(1, 8, 31, 40))


def test_bev_encoder_accepts_paper_sized_grid():
    from lss.models.bev_encoder import BEVEncoder

    model = BEVEncoder(in_channels=2).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 2, 200, 200))
    assert logits.shape == (1, 1, 200, 200)

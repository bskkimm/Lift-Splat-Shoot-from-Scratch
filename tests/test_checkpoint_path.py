import pytest, torch
from lss.checkpoint import load_checkpoint
def test_missing_checkpoint_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError): load_checkpoint(torch.nn.Linear(1,1),tmp_path/"missing.pt")

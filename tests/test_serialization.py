import torch
from lss.models.serialization import export_state_dict
def test_export_state_dict_writes_file(tmp_path):
    path=tmp_path/"model.pt"; export_state_dict(torch.nn.Linear(1,1),path); assert "state_dict" in torch.load(path)

import torch
from lss.models.serialization import export_state_dict
def test_export_state_dict_creates_parent_dirs(tmp_path):
    path=tmp_path/"nested"/"model.pt"; export_state_dict(torch.nn.Linear(1,1),path); assert path.exists()

import torch
from lss.models.serialization import export_state_dict
def test_state_export_leaves_no_temporary_file(tmp_path):
    path=tmp_path/"m.pt"; export_state_dict(torch.nn.Linear(1,1),path); assert not (tmp_path/"m.pt.tmp").exists()

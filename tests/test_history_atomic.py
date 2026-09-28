import torch
from lss.engine import fit
def test_history_has_no_temporary_file(tmp_path):
    m=torch.nn.Linear(1,1); o=torch.optim.SGD(m.parameters(),.1); fit(m,[((torch.ones(1,1),),torch.ones(1,1))],o,lambda x,y:((x-y)**2).mean(),1,checkpoint_dir=tmp_path); assert not (tmp_path/"history.json.tmp").exists()

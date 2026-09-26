import torch
from lss.engine import fit
def test_fit_writes_history(tmp_path):
    m=torch.nn.Linear(1,1); o=torch.optim.SGD(m.parameters(),.1); loader=[((torch.ones(1,1),),torch.ones(1,1))]; fit(m,loader,o,lambda x,y:((x-y)**2).mean(),1,checkpoint_dir=tmp_path); assert (tmp_path/"history.json").exists()

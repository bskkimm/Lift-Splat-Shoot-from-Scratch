import torch
from lss.engine import train_step
def test_train_step_accepts_gradient_clipping():
    model=torch.nn.Linear(1,1); opt=torch.optim.SGD(model.parameters(),.1); assert train_step(model,opt,(torch.ones(1,1),),lambda x:(x**2).mean(),1.0) >= 0

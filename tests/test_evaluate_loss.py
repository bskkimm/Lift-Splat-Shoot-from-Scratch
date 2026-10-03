import torch
from lss.engine import evaluate_loss
def test_evaluate_loss_averages_batches():
    model=torch.nn.Linear(1,1); loader=[((torch.ones(1,1),),torch.ones(1,1))]; assert evaluate_loss(model,loader,lambda x,y:((x-y)**2).mean()) >= 0

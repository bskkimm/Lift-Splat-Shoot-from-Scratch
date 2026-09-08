import torch
from lss.box_codec import decode_yaw, encode_yaw
def test_yaw_codec_round_trips():
    yaw=torch.tensor([-.5,0.,1.]); assert torch.allclose(decode_yaw(encode_yaw(yaw)),yaw)

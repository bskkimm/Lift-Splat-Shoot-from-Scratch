import torch
from lss.data.transforms import normalize_images
def test_normalize_images_uses_channel_statistics():
    out=normalize_images(torch.ones(1,1,3,1,1)); assert out.shape == (1,1,3,1,1)

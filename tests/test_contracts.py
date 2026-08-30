import pytest
import torch
from lss.data import CameraBatch


def test_camera_batch_validates_shapes():
    batch = CameraBatch(torch.zeros(2, 6, 3, 8, 8), torch.eye(3).expand(2, 6, 3, 3), torch.eye(4).expand(2, 6, 4, 4))
    assert batch.images.shape[:2] == (2, 6)


def test_camera_batch_rejects_wrong_image_rank():
    with pytest.raises(ValueError):
        CameraBatch(torch.zeros(2, 3, 8, 8), torch.zeros(2, 6, 3, 3), torch.zeros(2, 6, 4, 4))

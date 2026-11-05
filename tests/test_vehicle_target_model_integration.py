import torch

from lss.data.collate import collate_camera_samples
from lss.data.nuscenes_dataset import NuScenesCameraDataset
from lss.losses import binary_segmentation_loss
from lss.models.segmentation import LSSSegmentation
from tests.test_nuscenes_reference_frames import make_reference_fixture


def test_synthetic_raw_metadata_mask_drives_a_segmentation_optimizer_step(tmp_path):
    make_reference_fixture(tmp_path)
    dataset = NuScenesCameraDataset(dataroot=tmp_path, version="v1.0-mini",
                                    image_size=(32, 64),
                                    bev_bounds=((0, 8, 1), (-4, 4, 1)))
    sample = dataset[0]
    batch = collate_camera_samples([sample, sample])
    model = LSSSegmentation(x_bounds=(0, 8, 1), y_bounds=(-4, 4, 1),
                            z_bounds=(-5, 5, 10), depth_bounds=(1, 3, 1),
                            context_channels=8, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.bev_encoder.output[-1].weight.detach().clone()

    logits = model(batch["images"], batch["intrinsics"], batch["extrinsics"])
    loss = binary_segmentation_loss(logits, batch["bev_target"])
    loss.backward()
    optimizer.step()

    assert logits.shape == batch["bev_target"].shape == (2, 1, 8, 8)
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.bev_encoder.output[-1].weight)

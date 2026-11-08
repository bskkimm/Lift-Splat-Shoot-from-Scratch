import hashlib
from pathlib import Path

import pytest
import torch
from torchvision.models import efficientnet_b0

from lss.models.camera_encoder import CameraEncoder


@pytest.fixture(scope="module")
def backbone_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("backbone-init") / "weights.pt"
    torch.manual_seed(91)
    torch.save(efficientnet_b0(weights=None).state_dict(), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_torchvision_weights_initialize_all_camera_features(backbone_checkpoint):
    from lss.models.initialization import load_torchvision_b0_features

    path, digest = backbone_checkpoint
    encoder = CameraEncoder(2, 8, pretrained=False)
    projection_before = encoder.projection.weight.detach().clone()
    provenance = load_torchvision_b0_features(encoder, path, digest, weights_id="test-fixture")
    state = torch.load(path, map_location="cpu", weights_only=True)

    for key, value in encoder.features.state_dict().items():
        torch.testing.assert_close(value, state[f"features.{key}"], rtol=0, atol=0)
    torch.testing.assert_close(encoder.projection.weight, projection_before, rtol=0, atol=0)
    assert provenance["sha256"] == digest and provenance["weights_id"] == "test-fixture"
    assert provenance["path"] == str(Path(path).resolve())


def test_wrong_checksum_does_not_modify_encoder(backbone_checkpoint):
    from lss.models.initialization import load_torchvision_b0_features

    path, _ = backbone_checkpoint
    encoder = CameraEncoder(2, 8, pretrained=False)
    before = encoder.features[0][0].weight.detach().clone()
    with pytest.raises(ValueError, match="SHA-256"):
        load_torchvision_b0_features(encoder, path, "0" * 64, weights_id="fixture")
    torch.testing.assert_close(encoder.features[0][0].weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["missing", "shape", "unexpected"])
def test_incompatible_backbone_state_is_rejected_without_partial_loading(tmp_path, backbone_checkpoint, mutation):
    from lss.models.initialization import load_torchvision_b0_features

    state = torch.load(backbone_checkpoint[0], map_location="cpu", weights_only=True)
    key = "features.7.0.block.0.0.weight"
    if mutation == "missing":
        state.pop(key)
    elif mutation == "shape":
        state[key] = torch.zeros(1)
    else:
        state["features.7.unexpected"] = torch.zeros(1)
    path = tmp_path / "bad.pt"
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    encoder = CameraEncoder(2, 8, pretrained=False)
    before = {name: value.detach().clone() for name, value in encoder.features.state_dict().items()}

    with pytest.raises(ValueError, match="feature"):
        load_torchvision_b0_features(encoder, path, digest, weights_id="fixture")
    for name, value in encoder.features.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

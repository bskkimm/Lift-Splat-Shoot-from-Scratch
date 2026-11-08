"""Offline, checksum-verified initialization of the TorchVision image path."""

import hashlib
import hmac
import io
import re
from pathlib import Path

import torch


def load_torchvision_b0_features(encoder, path, sha256, weights_id):
    """Load verified TorchVision EfficientNet-B0 feature stages 0–7 only.

    ``path`` is a local, weights-only TorchVision state dictionary. The caller
    supplies its complete SHA-256 and a descriptive identifier. The hash is
    calculated over the same bytes passed to ``torch.load``. The unused
    TorchVision feature stage 8 and classifier are ignored; the local fusion,
    depth and context parameters are not initialized by this helper. A digest
    confirms bytes, not the accuracy of the caller's provenance statement.
    """
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise ValueError("SHA-256 must contain exactly 64 hexadecimal characters")
    if not isinstance(weights_id, str) or not weights_id.strip():
        raise ValueError("weights_id must be a nonempty string")
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if not hmac.compare_digest(actual, sha256.lower()):
        raise ValueError("SHA-256 mismatch for camera backbone weights")
    state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("feature checkpoint must be a state dictionary")
    expected = encoder.features.state_dict()
    selected = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise ValueError("feature checkpoint keys must be strings")
        if key.startswith("classifier.") or key.startswith("features.8."):
            continue
        if not key.startswith("features."):
            raise ValueError(f"unexpected feature checkpoint key: {key}")
        selected[key[len("features."):]] = value
    if selected.keys() != expected.keys():
        missing = sorted(expected.keys() - selected.keys())
        unexpected = sorted(selected.keys() - expected.keys())
        raise ValueError(f"feature checkpoint keys do not match: missing={missing}, unexpected={unexpected}")
    for key, value in expected.items():
        candidate = selected[key]
        if not torch.is_tensor(candidate) or candidate.shape != value.shape or candidate.dtype != value.dtype:
            raise ValueError(f"feature checkpoint tensor shape or dtype mismatch: {key}")
    encoder.features.load_state_dict(selected, strict=True)
    return {"path": str(path), "sha256": actual, "weights_id": weights_id,
            "scope": "torchvision-efficientnet-b0-features-0-7"}

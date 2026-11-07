import json
import os

import pytest
import torch

from tests.test_nuscenes_official_splits import make_two_scene_dataset
from tests.test_practical_training_cli import run_training


@pytest.mark.skipif(os.environ.get("LSS_TEST_CUDA") != "1", reason="opt-in CUDA integration test")
def test_cuda_training_cli_configures_workspace_and_saves_rng(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.fail("LSS_TEST_CUDA=1 but CUDA is unavailable")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    output = tmp_path / "run"

    result = run_training(data, output, epochs=1, extra_args=("--device", "cuda"))

    assert result.returncode == 0, result.stderr
    state = torch.load(output / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    config = json.loads(state["config_json"])
    assert config["device"] == "cuda"
    assert config["cublas_workspace_config"] == ":4096:8"
    assert state["cuda_rng"] is not None

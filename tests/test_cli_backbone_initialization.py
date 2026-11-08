import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torchvision.models import efficientnet_b0

from tests.test_nuscenes_official_splits import make_two_scene_dataset
from tests.test_practical_evaluation_cli import run_evaluation
from tests.test_practical_training_cli import run_training


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli-init") / "weights.pt"
    torch.save(efficientnet_b0(weights=None).state_dict(), path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def run_dry(*args):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run([sys.executable, str(ROOT / "train.py"), "--dry-run", *args],
                          capture_output=True, text=True, env=env, timeout=60)


def test_dry_run_loads_verified_local_backbone_without_writing(weights, tmp_path):
    path, digest = weights
    output = tmp_path / "output"
    result = run_dry("--output-dir", str(output), "--backbone-weights", str(path),
                     "--backbone-sha256", digest, "--backbone-weights-id", "synthetic-fixture")
    assert result.returncode == 0, result.stderr
    assert not output.exists()


def test_dry_run_rejects_incomplete_or_incorrect_weight_arguments(weights):
    path, _ = weights
    incomplete = run_dry("--backbone-weights", str(path))
    assert incomplete.returncode != 0
    assert "together" in incomplete.stderr.lower()
    wrong = run_dry("--backbone-weights", str(path), "--backbone-sha256", "0" * 64,
                    "--backbone-weights-id", "synthetic-fixture")
    assert wrong.returncode != 0
    assert "sha-256" in wrong.stderr.lower()


def test_training_checkpoint_records_verified_initialization_and_resumes(tmp_path, weights):
    path, digest = weights
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    output = tmp_path / "run"
    options = ("--backbone-weights", str(path), "--backbone-sha256", digest,
               "--backbone-weights-id", "synthetic-fixture")
    first = run_training(data, output, epochs=1, extra_args=options)
    assert first.returncode == 0, first.stderr
    checkpoint = output / "last_checkpoint.pt"
    config = json.loads(torch.load(checkpoint, map_location="cpu", weights_only=True)["config_json"])
    assert config["backbone_initialization"]["sha256"] == digest
    assert config["backbone_initialization"]["weights_id"] == "synthetic-fixture"
    second = run_training(data, output, epochs=2, resume=checkpoint, extra_args=options)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout.strip().splitlines()[-1])["epoch"] == 2
    evaluation = run_evaluation(checkpoint, "--dataroot", str(data))
    assert evaluation.returncode == 0, evaluation.stderr
    assert json.loads(evaluation.stdout.strip().splitlines()[-1])["checkpoint_epoch"] == 2

    before = checkpoint.read_bytes()
    wrong_options = ("--backbone-weights", str(path), "--backbone-sha256", "0" * 64,
                     "--backbone-weights-id", "synthetic-fixture")
    rejected = run_training(data, output, epochs=3, resume=checkpoint, extra_args=wrong_options)
    assert rejected.returncode != 0
    assert "sha-256" in rejected.stderr.lower()
    assert checkpoint.read_bytes() == before

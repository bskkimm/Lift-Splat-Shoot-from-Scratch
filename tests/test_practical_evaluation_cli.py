import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.test_nuscenes_official_splits import make_two_scene_dataset
from tests.test_practical_training_cli import run_training


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("practical-eval")
    data = root / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    output = root / "train"
    result = run_training(data, output, epochs=1)
    assert result.returncode == 0, result.stderr
    return data, output / "last_checkpoint.pt", json.loads(result.stdout.strip().splitlines()[-1])


def run_evaluation(checkpoint, *args):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return subprocess.run([sys.executable, str(ROOT / "eval.py"), "--checkpoint", str(checkpoint), *args],
                          capture_output=True, text=True, env=env, timeout=90)


def test_evaluation_runs_checkpoint_on_validation_split(trained_fixture, tmp_path):
    data, checkpoint, training_metrics = trained_fixture
    output = tmp_path / "metrics.json"

    result = run_evaluation(checkpoint, "--dataroot", str(data), "--output", str(output))

    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert json.loads(output.read_text()) == metrics
    assert metrics["metric_name"] == "practical_bev_vehicle_iou"
    assert metrics["scope"] == "full_split"
    assert metrics["split"] == "val" and metrics["checkpoint_epoch"] == 1
    assert metrics["metrics"] == training_metrics["validation"]


def test_evaluation_labels_requested_subset(trained_fixture):
    data, checkpoint, _ = trained_fixture
    result = run_evaluation(checkpoint, "--dataroot", str(data), "--max-samples", "1")
    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["scope"] == "subset"
    assert metrics["metrics"]["samples"] == 1


def test_incompatible_checkpoint_fails_without_result_file(tmp_path):
    checkpoint = tmp_path / "not-practical.pt"
    torch.save({"weights": torch.ones(1)}, checkpoint)
    output = tmp_path / "metrics.json"
    result = run_evaluation(checkpoint, "--output", str(output))
    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower()
    assert not output.exists()


def test_checkpoint_missing_configuration_is_rejected_clearly(tmp_path):
    checkpoint = tmp_path / "incomplete.pt"
    torch.save({"format_version": 1, "types": {"model": "lss.models.segmentation.LSSSegmentation"},
                "config_json": json.dumps({"recipe": "practical-resize-fp32-v1"})}, checkpoint)
    output = tmp_path / "metrics.json"
    result = run_evaluation(checkpoint, "--output", str(output))
    assert result.returncode != 0
    assert "missing checkpoint fields" in result.stderr.lower()
    assert not output.exists()


def test_evaluation_does_not_overwrite_existing_metrics(trained_fixture, tmp_path):
    data, checkpoint, _ = trained_fixture
    output = tmp_path / "metrics.json"
    output.write_bytes(b"existing")
    result = run_evaluation(checkpoint, "--dataroot", str(data), "--output", str(output))
    assert result.returncode != 0
    assert "exists" in result.stderr.lower()
    assert output.read_bytes() == b"existing"


def test_evaluation_rejects_nonpositive_subset_limit(trained_fixture):
    _, checkpoint, _ = trained_fixture
    result = run_evaluation(checkpoint, "--max-samples", "0")
    assert result.returncode != 0
    assert "max-samples" in result.stderr.lower()

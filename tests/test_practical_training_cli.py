import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from tests.test_nuscenes_official_splits import make_two_scene_dataset
from tests.test_nuscenes_raw_keyframes import make_raw_dataset, write_table


ROOT = Path(__file__).resolve().parents[1]


def run_training(data, output, epochs, resume=None, extra_args=()):
    command = [
        sys.executable, str(ROOT / "train.py"), "--dataroot", str(data),
        "--version", "v1.0-mini", "--output-dir", str(output),
        "--epochs", str(epochs), "--batch-size", "1", "--seed", "9",
        "--device", "cpu", "--image-size", "32", "64",
        "--x-bound", "-8", "8", "1", "--y-bound", "-8", "8", "1",
        "--z-bound", "-5", "5", "10", "--depth-bound", "1", "3", "1",
        "--context-channels", "8",
    ]
    if resume is not None:
        command += ["--resume", str(resume)]
    command += list(extra_args)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=90)


def test_training_cli_runs_real_optimization_and_validation(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    output = tmp_path / "run"

    result = run_training(data, output, epochs=1)

    assert result.returncode == 0, result.stderr
    checkpoint = torch.load(output / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["optimizer"]["state"]
    config = json.loads(checkpoint["config_json"])
    assert config["cpu_threads"] == 1
    assert config["deterministic_algorithms"] is True
    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["epoch"] == 1
    assert metrics["train"]["samples"] == 1 and metrics["validation"]["samples"] == 1


def test_cli_epoch_boundary_resume_matches_uninterrupted_training(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"

    full_result = run_training(data, full, epochs=2)
    first_result = run_training(data, interrupted, epochs=1)
    resume_result = run_training(data, interrupted, epochs=2, resume=interrupted / "last_checkpoint.pt")

    assert full_result.returncode == 0, full_result.stderr
    assert first_result.returncode == 0, first_result.stderr
    assert resume_result.returncode == 0, resume_result.stderr
    expected = torch.load(full / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    actual = torch.load(interrupted / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    assert expected["epoch"] == actual["epoch"] == 2
    for key, value in expected["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)
        else:
            assert actual["model"][key] == value
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    for key, state in expected["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(actual["optimizer"]["state"][key][name], value, rtol=0, atol=0)
    torch.testing.assert_close(actual["torch_rng"], expected["torch_rng"], rtol=0, atol=0)
    torch.testing.assert_close(actual["loader_rng"], expected["loader_rng"], rtol=0, atol=0)
    assert json.loads(full_result.stdout.strip().splitlines()[-1]) == json.loads(resume_result.stdout.strip().splitlines()[-1])


def test_independent_seeded_runs_match_at_first_epoch(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = run_training(data, first_dir, epochs=1)
    second = run_training(data, second_dir, epochs=1)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_state = torch.load(first_dir / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    second_state = torch.load(second_dir / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    for key, value in first_state["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(second_state["model"][key], value, rtol=0, atol=0)
        else:
            assert second_state["model"][key] == value
    torch.testing.assert_close(first_state["torch_rng"], second_state["torch_rng"], rtol=0, atol=0)
    torch.testing.assert_close(first_state["loader_rng"], second_state["loader_rng"], rtol=0, atol=0)


def test_cli_rejects_resume_with_changed_configuration(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    output = tmp_path / "run"
    first = run_training(data, output, epochs=1)
    assert first.returncode == 0, first.stderr
    checkpoint = output / "last_checkpoint.pt"
    before = checkpoint.read_bytes()

    result = run_training(data, output, epochs=2, resume=checkpoint, extra_args=("--lr", "0.002"))

    assert result.returncode != 0
    assert "config" in result.stderr.lower()
    assert checkpoint.read_bytes() == before


def test_training_cli_requires_dataroot_without_dry_run(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run([sys.executable, str(ROOT / "train.py"), "--output-dir", str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "dataroot" in result.stderr.lower()
    assert not output.exists()


def test_training_cli_rejects_empty_validation_split_before_writing(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_raw_dataset(data)
    write_table(data / "v1.0-mini", "scene", [{"token": "scene", "name": "scene-0061"}])
    output = tmp_path / "run"

    result = run_training(data, output, epochs=1)

    assert result.returncode != 0
    assert "validation split contains no samples" in result.stderr.lower()
    assert not output.exists()


def test_fresh_run_does_not_overwrite_existing_checkpoint(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    output = tmp_path / "run"
    output.mkdir()
    checkpoint = output / "last_checkpoint.pt"
    checkpoint.write_bytes(b"existing checkpoint")

    result = run_training(data, output, epochs=1)

    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower() and "exists" in result.stderr.lower()
    assert checkpoint.read_bytes() == b"existing checkpoint"


def test_resume_does_not_overwrite_a_different_existing_checkpoint(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    source = tmp_path / "source"
    first = run_training(data, source, epochs=1)
    assert first.returncode == 0, first.stderr
    output = tmp_path / "other"
    output.mkdir()
    checkpoint = output / "last_checkpoint.pt"
    checkpoint.write_bytes(b"other checkpoint")

    result = run_training(data, output, epochs=2, resume=source / "last_checkpoint.pt")

    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower() and "exists" in result.stderr.lower()
    assert checkpoint.read_bytes() == b"other checkpoint"

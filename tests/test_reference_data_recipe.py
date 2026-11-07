import json

import torch

from tests.test_nuscenes_official_splits import make_two_scene_dataset
from tests.test_practical_evaluation_cli import run_evaluation
from tests.test_practical_training_cli import run_training


REFERENCE_ARGS = (
    "--recipe", "reference-data", "--camera-count", "1",
    "--resize-limits", "32", "32", "--bottom-limits", "0", "0",
    "--rotation-limits", "0", "0", "--no-random-flip",
)


def test_reference_data_recipe_trains_and_evaluates_matching_validation(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    output = tmp_path / "run"

    train = run_training(data, output, epochs=1, extra_args=REFERENCE_ARGS)

    assert train.returncode == 0, train.stderr
    checkpoint = output / "last_checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = json.loads(state["config_json"])
    assert config["recipe"] == "reference-data-fp32-v1"
    assert config["camera_count"] == 1
    assert config["augmentation"]["resize_limits"] == [32.0, 32.0]
    evaluation = run_evaluation(checkpoint, "--dataroot", str(data))
    assert evaluation.returncode == 0, evaluation.stderr
    train_metrics = json.loads(train.stdout.strip().splitlines()[-1])["validation"]
    eval_metrics = json.loads(evaluation.stdout.strip().splitlines()[-1])["metrics"]
    assert train_metrics == eval_metrics


def test_reference_data_recipe_resumes_at_epoch_boundary(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    make_two_scene_dataset(data)
    output = tmp_path / "run"
    full_output = tmp_path / "full"
    stochastic_args = ("--recipe", "reference-data", "--camera-count", "1",
                       "--resize-limits", "24", "32", "--bottom-limits", "0", "0.2",
                       "--rotation-limits", "-5", "5")
    full = run_training(data, full_output, epochs=2, extra_args=stochastic_args)
    assert full.returncode == 0, full.stderr
    first = run_training(data, output, epochs=1, extra_args=stochastic_args)
    assert first.returncode == 0, first.stderr
    second = run_training(data, output, epochs=2, resume=output / "last_checkpoint.pt",
                          extra_args=stochastic_args)
    assert second.returncode == 0, second.stderr
    assert json.loads(second.stdout.strip().splitlines()[-1])["epoch"] == 2
    expected = torch.load(full_output / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    actual = torch.load(output / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    for key, value in expected["model"].items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)
        else:
            assert actual["model"][key] == value

import subprocess
import sys
from pathlib import Path

import torch

from lss.models.lss import LSS


ROOT = Path(__file__).resolve().parents[1]


def test_training_without_dry_run_fails_without_saving_untrained_weights(tmp_path):
    output_dir = tmp_path / "checkpoints"
    result = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--output-dir", str(output_dir)],
        capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert "not implemented" in result.stderr.lower()
    assert not output_dir.exists()


def test_evaluation_fails_without_writing_empty_predictions(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "predictions.json"
    torch.save(LSS().state_dict(), checkpoint)
    result = subprocess.run(
        [sys.executable, str(ROOT / "eval.py"), "--checkpoint", str(checkpoint),
         "--output", str(output)],
        capture_output=True, text=True,
    )

    assert result.returncode != 0
    assert "not implemented" in result.stderr.lower()
    assert not output.exists()

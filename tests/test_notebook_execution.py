import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_notebook(path):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "execute_notebook.py"), str(path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )


def test_walkthrough_executes_offline_without_modifying_notebook():
    notebook = ROOT / "notebooks" / "lss_e2e_walkthrough.ipynb"
    before = notebook.read_bytes()
    result = run_notebook(notebook)
    assert result.returncode == 0, result.stderr
    assert "NOTEBOOK_EXECUTION_COMPLETE" in result.stdout
    assert notebook.read_bytes() == before


def test_walkthrough_optional_checkpoint_fails_clearly_when_missing(tmp_path):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", LSS_NOTEBOOK_CHECKPOINT="1",
               LSS_NOTEBOOK_CHECKPOINT_PATH=str(tmp_path / "missing.pt"))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "execute_notebook.py"),
         str(ROOT / "notebooks" / "lss_e2e_walkthrough.ipynb")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "checkpoint file not found" in result.stderr


def test_walkthrough_optional_real_data_fails_clearly_when_missing(tmp_path):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", LSS_NOTEBOOK_REAL_DATA="1",
               LSS_NOTEBOOK_DATAROOT=str(tmp_path / "missing"))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "execute_notebook.py"),
         str(ROOT / "notebooks" / "lss_e2e_walkthrough.ipynb")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "nuScenes version directory not found" in result.stderr


def test_walkthrough_optional_parity_runs_on_synthetic_inputs():
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg", LSS_NOTEBOOK_PARITY="1")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "execute_notebook.py"),
         str(ROOT / "notebooks" / "lss_e2e_walkthrough.ipynb")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "synthetic notebook/package output parity passed" in result.stdout


def test_walkthrough_loads_compatible_practical_checkpoint(tmp_path):
    import torch
    from lss.models.segmentation import LSSSegmentation
    from lss.training_checkpoint import save_epoch_checkpoint

    torch.set_num_threads(1)
    config = {"x_bounds": (-8, 8, 1), "y_bounds": (-8, 8, 1), "z_bounds": (0, 3, 3),
              "depth_bounds": (1, 3, 1), "context_channels": 8, "out_channels": 1,
              "recipe": "practical-resize-fp32-v1"}
    model = LSSSegmentation(**{key: value for key, value in config.items() if key != "recipe"})
    path = tmp_path / "checkpoint.pt"
    save_epoch_checkpoint(path, model, torch.optim.Adam(model.parameters()), 2, config, torch.Generator())
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg",
               LSS_NOTEBOOK_CHECKPOINT="1", LSS_NOTEBOOK_CHECKPOINT_PATH=str(path))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "execute_notebook.py"),
         str(ROOT / "notebooks" / "lss_e2e_walkthrough.ipynb")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "loaded practical checkpoint epoch 2" in result.stdout


def test_notebook_runner_rejects_network_calls(tmp_path):
    notebook = tmp_path / "network.ipynb"
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": [
        "import socket\n", "socket.create_connection(('example.com', 80))\n"
    ]}]}))
    result = run_notebook(notebook)
    assert result.returncode != 0
    assert "network access disabled" in result.stderr


def test_notebook_runner_reports_failing_cell(tmp_path):
    notebook = tmp_path / "broken.ipynb"
    notebook.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": []},
                                               {"cell_type": "code", "source": ["raise ValueError('probe')"]}]}))
    result = run_notebook(notebook)
    assert result.returncode != 0
    assert "cell 1" in result.stderr and "probe" in result.stderr

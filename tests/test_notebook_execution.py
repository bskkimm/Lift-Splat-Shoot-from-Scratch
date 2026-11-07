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

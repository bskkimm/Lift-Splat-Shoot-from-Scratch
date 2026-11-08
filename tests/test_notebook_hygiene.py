import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check_notebook_clean.py"


@pytest.mark.parametrize("cell, expected", [
    ({"cell_type": "code", "execution_count": 1, "outputs": []}, "execution_count"),
    ({"cell_type": "code", "execution_count": None, "outputs": [{"output_type": "stream"}]}, "outputs"),
    ({"cell_type": "code", "execution_count": None}, "outputs"),
])
def test_hygiene_rejects_notebook_with_execution_artifacts(tmp_path, cell, expected):
    path = tmp_path / "example.ipynb"
    path.write_text(json.dumps({"cells": [cell]}))
    result = subprocess.run([sys.executable, str(CHECK), str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert expected in result.stderr


def test_hygiene_accepts_clean_notebook_and_reports_invalid_json(tmp_path):
    good = tmp_path / "good.ipynb"
    good.write_text(json.dumps({"cells": [{"cell_type": "code", "execution_count": None, "outputs": []}]}))
    assert subprocess.run([sys.executable, str(CHECK), str(good)], capture_output=True).returncode == 0
    bad = tmp_path / "bad.ipynb"
    bad.write_text("{")
    result = subprocess.run([sys.executable, str(CHECK), str(bad)], capture_output=True, text=True)
    assert result.returncode != 0 and "bad.ipynb" in result.stderr

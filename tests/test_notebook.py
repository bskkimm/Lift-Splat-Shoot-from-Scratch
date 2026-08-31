import json
from pathlib import Path
def test_walkthrough_is_unexecuted_notebook():
    notebook = json.loads(Path("notebooks/lss_e2e_walkthrough.ipynb").read_text())
    assert all(cell["execution_count"] is None for cell in notebook["cells"] if cell["cell_type"] == "code")

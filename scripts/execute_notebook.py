"""Execute Python notebook cells in this process without writing notebook output.

This lightweight smoke runner rejects Python socket connections. It is not an
operating-system sandbox and must only be used with trusted notebooks.
"""

import json
import os
import socket
import sys
from pathlib import Path


def _offline(*args, **kwargs):
    raise RuntimeError("network access disabled during notebook smoke execution")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: execute_notebook.py NOTEBOOK.ipynb")
    notebook_path = Path(sys.argv[1]).resolve()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    # This prevents common accidental Python-level network usage, not malicious code.
    socket.socket.connect = _offline
    socket.socket.connect_ex = _offline
    socket.create_connection = _offline
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    namespace = {"__name__": "__main__"}
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        try:
            exec(compile(source, f"{notebook_path}:cell {index}", "exec"), namespace)
        except Exception as exc:
            raise RuntimeError(f"notebook cell {index} failed") from exc
        count += 1
    if count == 0:
        raise ValueError("notebook has no executable code cells")
    print(f"NOTEBOOK_EXECUTION_COMPLETE: {count} code cells")


if __name__ == "__main__":
    main()

"""Reject notebooks with stored execution counts or outputs; never modify files."""

import json
import sys
from pathlib import Path


def check_notebook(path):
    """Return diagnostic strings for one notebook."""
    try:
        notebook = json.loads(Path(path).read_text(encoding="utf-8"))
        cells = notebook["cells"]
        if not isinstance(cells, list):
            raise ValueError("cells must be a list")
        errors = []
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                errors.append(f"{path}: cell {index}: cell must be an object")
            elif cell.get("cell_type") == "code":
                if "execution_count" not in cell or cell["execution_count"] is not None:
                    errors.append(f"{path}: cell {index}: execution_count must be null")
                if cell.get("outputs") != []:
                    errors.append(f"{path}: cell {index}: outputs must be an empty list")
        return errors
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"{path}: invalid notebook: {exc}"]


def main(paths):
    errors = [error for path in paths for error in check_notebook(path)]
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

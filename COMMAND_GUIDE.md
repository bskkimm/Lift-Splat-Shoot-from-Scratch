# Command Guide

Run `python train.py --help` or `python eval.py --help` for the current CLI.

Run the complete synthetic test suite with `python3 -m pytest -q`.

The suite is also runnable through `pytest -q` after installing the package in
editable mode with `pip install -e .`.

Use `python train.py --dry-run --seed 0` to validate model construction, and
`python eval.py --help` to inspect checkpoint evaluation and JSON export.

Example checkpoint flow:

```bash
python train.py --dry-run --output-dir outputs/smoke
python eval.py --help
```

Prediction exports use the nuScenes-compatible structure `{\"results\": [],
\"meta\": {\"version\": \"v1.0-trainval\"}}`.

Use `lss.models.diagnostics.parameter_count(model)` to inspect model size.

Training optimizer defaults are `--lr 2e-4 --weight-decay 0.01`.

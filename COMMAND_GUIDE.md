# Command guide

These commands describe the current scaffold. They are not a paper-reproduction
workflow; see [the audit](docs/paper-to-code.md) for known gaps.

## Tests

```bash
python3 -m pytest -q
```

The tests use synthetic inputs and do not establish nuScenes or paper parity.

## Model construction smoke check

```bash
python3 train.py --dry-run --seed 0
```

This constructs a model without saving it or performing optimization. Without
`--dry-run`, the command exits with an error because the training loop is not
wired up. Options such as `--epochs`, `--lr`, and `--scheduler` do not currently
control a training loop.

## CLI help

```bash
python3 train.py --help
python3 eval.py --help
```

Evaluation exits with an error until inference and the evaluation protocol are
wired up. It does not load checkpoints or write prediction files.

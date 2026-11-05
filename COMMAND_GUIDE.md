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

This constructs a model without saving it. Despite the training-oriented output,
it performs no optimization. Without `--dry-run`, the script saves **untrained**
initial weights; options such as `--epochs`, `--lr`, and `--scheduler` do not
currently control a training loop.

## CLI help

```bash
python3 train.py --help
python3 eval.py --help
```

The evaluation command currently only loads a checkpoint. When `--output` is
specified, it writes an empty placeholder rather than running inference or the
nuScenes evaluation protocol. Do not submit that file as a prediction result.

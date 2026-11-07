# Command guide

These commands describe the current practical workflow. They are not a
paper-reproduction recipe; see [the audit](docs/paper-to-code.md) for known gaps.

## Tests

```bash
python3 -m pytest -q
```

The tests use synthetic inputs and do not establish nuScenes or paper parity.

## Model construction smoke check

```bash
python3 train.py --dry-run --seed 0
```

This constructs the segmentation model without reading the dataset, saving a
checkpoint, or performing optimization.

## Practical segmentation training

With nuScenes metadata and images available, the command below selects the
official mini train/validation scene lists, trains with Adam, validates and
writes an epoch-boundary checkpoint:

```bash
python3 train.py --dataroot /path/to/nuscenes --version v1.0-mini \
  --epochs 1 --batch-size 1 --device cpu --output-dir outputs/practical-mini
```

This practical recipe uses FP32, direct image resize, random image-encoder
initialization, no paper augmentation or camera dropout, and zero data-loader
workers. It enables PyTorch deterministic algorithms and defaults to one CPU
thread (`--cpu-threads 1`); unsupported deterministic operations fail rather
than silently switching algorithms. It is not the paper training recipe. Training and validation emit JSON
statistics per epoch; training IoU is diagnostic, not an official metric.
The command has synthetic fixture coverage but has not been run on full nuScenes.
The current loader reads full metadata tables separately for train and validation,
which can require substantial host memory; full-dataset sizing is unmeasured.

Resume at a completed epoch boundary with the **same configuration** and a
larger total target epoch count, for example:

```bash
python3 train.py --dataroot /path/to/nuscenes --version v1.0-mini \
  --epochs 2 --batch-size 1 --device cpu --output-dir outputs/practical-mini \
  --resume outputs/practical-mini/last_checkpoint.pt
```

CPU, one-thread/zero-worker resume equivalence has synthetic test coverage; full-model
real-data, GPU and multiworker recovery remain unverified. Configuration checks
include the dataset path and version, not a hash of the dataset contents.
Fresh runs refuse to overwrite an existing checkpoint; choose a new output path
or resume that same checkpoint.

## CLI help

```bash
python3 train.py --help
python3 eval.py --help
```

The standalone evaluation CLI exits with an error until checkpoint inference and
the evaluation protocol are wired up. It does not write prediction files.

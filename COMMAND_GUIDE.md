# Command guide

These commands describe the current practical workflow. They are not a
paper-reproduction recipe; see [the audit](docs/paper-to-code.md) for known gaps.

## Tests

```bash
python3 -m pytest -q
```

Run repository hygiene checks with `pre-commit run --all-files` after installing
`pre-commit==4.3.0`; formatting hooks can modify files when a check fails.

The tests use synthetic inputs and do not establish nuScenes or paper parity.
The current notebook's cells also run in a fresh subprocess as part of the
suite. For a standalone smoke check, run:

```bash
python3 scripts/execute_notebook.py notebooks/lss_e2e_walkthrough.ipynb
```

The runner disables Python socket connections and hides CUDA devices, but is
not an operating-system sandbox. Run only trusted notebooks. The current
notebook locally defines the practical segmentation model and runs a synthetic
optimizer step. Its package parity tests compare evaluation outputs and gradients,
plus training loss, gradients and post-update state with matched weights and RNG.
Real-data and pretrained checkpoint demonstrations remain future work.

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
than silently switching algorithms. For CUDA, the CLI sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization if unset; an
existing `:16:8` or `:4096:8` is preserved. It is not the paper training recipe.
Training and validation emit JSON
statistics per epoch; training IoU is diagnostic, not an official metric.
The command has synthetic fixture coverage but has not been run on full nuScenes.
The current loader reads full metadata tables separately for train and validation,
which can require substantial host memory; full-dataset sizing is unmeasured.

`--recipe reference-data` enables reference-oriented image augmentation and
five randomly selected training cameras by default; validation uses every
available camera and a deterministic crop. It retains random TorchVision
initialization and the current ego-motion compensation, so it is **not** a
faithful reference reproduction. The policy has only synthetic fixture coverage.
Use `--camera-count`, `--resize-limits`, `--bottom-limits`, `--rotation-limits`,
and `--no-random-flip` to change the recorded recipe configuration.

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

To run the opt-in synthetic CUDA CLI smoke test on a CUDA-capable system:

```bash
LSS_TEST_CUDA=1 python3 -m pytest -q tests/test_practical_training_cuda.py
```

The test fails if explicitly enabled without CUDA. It checks one synthetic
training/validation epoch and CUDA RNG checkpointing, not GPU resume parity.

## CLI help

```bash
python3 train.py --help
python3 eval.py --help
```

## Practical checkpoint evaluation

Evaluate a checkpoint on its configured validation split:

```bash
python3 eval.py --checkpoint outputs/practical-mini/last_checkpoint.pt \
  --output outputs/practical-mini/validation.json
```

The result is labeled `practical_bev_vehicle_iou`, includes raw intersection and
union counts, and does not claim paper or official metric parity. To limit the
number of validation samples, add `--max-samples N`; results are then labeled as
subset diagnostics. The data root defaults to the checkpoint's configured path
and may be replaced with `--dataroot`. Existing output files are not overwritten.
The CLI has synthetic fixture coverage but has not been validated on full nuScenes.

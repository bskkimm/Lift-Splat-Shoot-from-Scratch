# Real nuScenes validation-loader smoke

## Result and scope

On 2026-11-08, the opt-in test `tests/test_real_nuscenes_loader.py` passed
against a locally available nuScenes `v1.0-trainval` layout. The dataset loader
selected **6,019** validation samples using the nuScenes devkit scene list.
The first returned sample contained six images of shape `[6,3,128,352]`, six
intrinsic and extrinsic matrices, and a binary vehicle target of shape
`[1,200,200]`; the test checked finite images and matrices.

This is a **data-only smoke test**, not a comparison with reference masks or
geometry, a training run, an IoU measurement, or a paper reproduction result.
It does not verify every sample or confirm image availability across the split.
The dataset was not hashed; exact asset identity beyond version/layout was not
independently established. No dataset files or sample content are committed.

## Command and environment

Code was based on commit `33a944450d1f57be666538b81bc4d7d74433942e`, plus the
new uncommitted test that is included with this report. The command was run in
`torch_env` on CPU with Python 3.11.14, PyTorch 2.9.1+cu130, TorchVision
0.24.1+cu130, NumPy 1.26.4, OpenCV 4.11.0 and nuScenes devkit 1.2.0. The host
reported an Intel Core i9-14900K and 32 logical CPUs. No GPU was used.

```bash
ulimit -v 16777216
/usr/bin/time -v conda run -n torch_env env \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  LSS_REAL_NUSCENES_ROOT=/path/to/nuscenes \
  python -m pytest -q tests/test_real_nuscenes_loader.py
```

The test passed twice; the timed second invocation reported 48.52 seconds wall
time and 8,622,564 KiB maximum resident set size for the command's process tree
as reported by `/usr/bin/time`. These numbers include Python startup, loading
metadata and a single sample, not training throughput. Cache conditions were
not controlled, so they are not a benchmark. The 16 GiB virtual-memory limit
was used for this single test; the training CLI currently constructs separate
train and validation datasets, whose combined memory use was not measured.

The real-data test is skipped unless `LSS_REAL_NUSCENES_ROOT` is set; when set,
missing files, wrong split counts or incompatible sample shapes fail the test.

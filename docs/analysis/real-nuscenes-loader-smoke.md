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

## Streaming sample-data follow-up

On the same host and environment, the opt-in test passed after changing
`sample_data.json` loading to stream and retain selected keyframes (base code
commit `8c32bd59e35e4866687e432573713a0e6f0b5509`, plus the then-uncommitted
streaming change). The timed
invocation of the same command reported 47.24 seconds wall and 5,326,780 KiB
maximum RSS. The test again checked only the validation split count and one
sample's shape/finite/binary properties, not numerical parity with the earlier
run. This single before/after observation is not a controlled throughput or
memory benchmark: cache and other host load were not controlled. Remaining
metadata tables are still loaded eagerly and separate train/validation dataset
memory remains unmeasured.

## Independent one-sample mask comparison

On the same host and environment, an opt-in test compared the first validation
sample's generated vehicle mask against a separately constructed mask using
`nuscenes.utils.data_classes.Box`, `pyquaternion`, raw metadata and OpenCV
polygon filling. The test required at least one vehicle and asserted exact
pixel equality; it passed. It uses the same rounded-grid and OpenCV raster
convention, so this is not an independent assessment of that convention itself.

```bash
ulimit -v 16777216
/usr/bin/time -v conda run -n torch_env env \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  LSS_REAL_NUSCENES_ROOT=/path/to/nuscenes \
  python -m pytest -q tests/test_real_nuscenes_mask_parity.py
```

The command used code at base `1871fe657e085adea8bf54fe95a9c351bfda8d94`
plus the then-uncommitted test included with this report. It took 47.03 seconds
wall and reported 5,323,588 KiB maximum RSS. This single observation is not a
performance benchmark. It does not establish parity across the full dataset,
validate camera projection, prove checkpoint parity, or measure any model metric.

## Independent one-sample camera-transform comparison

On the same host and software environment, the first validation sample's six
camera-to-reference-ego rotations and translations were compared with an
independent NumPy/`pyquaternion` composition of the raw calibration, camera-time
ego pose and LiDAR-time ego pose. Resized intrinsic matrices were also checked
against the raw matrices and sample-data image dimensions. The test passed with
absolute tolerances of `1e-5` for rotations and `1e-3` metres for translations,
and `rtol=1e-6, atol=1e-5` for intrinsics.

```bash
ulimit -v 16777216
/usr/bin/time -v conda run -n torch_env env \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  LSS_REAL_NUSCENES_ROOT=/path/to/nuscenes \
  python -m pytest -q tests/test_real_nuscenes_camera_parity.py
```

The command used code at base `1a81ea436fc11541e07d2f6c8e92db88596dbf51`
plus the then-uncommitted test included with this report. It took 44.68 seconds
wall and reported 5,327,004 KiB maximum RSS. The composition validates the
repository's timestamp-compensated convention for one sample; it does not
prove numerical equivalence to the original LSS implementation, which uses
camera calibrated-sensor transforms directly, or cover the full dataset.

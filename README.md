# Lift-Splat-Shoot

Pure PyTorch implementation of Lift-Splat-Shoot (LSS), built incrementally
with small synthetic tests and configurable model defaults.

## Setup

```bash
pip install -e .
pytest -q
```

The implementation uses camera images, calibration matrices, a learned depth
distribution, geometric lifting, and BEV voxel pooling. The original-style
defaults are available through `lss.config.OfficialConfig`.

## Commands

```bash
python train.py --help
python eval.py --help
```

Real nuScenes evaluation additionally requires `nuscenes-devkit`.

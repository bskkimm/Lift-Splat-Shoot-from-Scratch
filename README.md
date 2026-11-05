# Lift-Splat-Shoot in PyTorch

This repository is being rebuilt as an educational, from-scratch PyTorch
implementation of [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711).
The goal is a transparent implementation of multi-camera bird's-eye-view (BEV)
segmentation and trajectory scoring, without OpenMMLab model or training
frameworks. See [the paper-to-code audit](docs/paper-to-code.md) for the source
references, planned components, and known deviations.

## Current status

**The current code is an experimental scaffold, not a faithful LSS reproduction.**
In particular:

- `train.py --dry-run` constructs a model without training. Other invocations
  fail explicitly because training and nuScenes data loading are not wired up.
- `eval.py` fails explicitly because inference and evaluation are not wired up;
  it does not generate predictions or official nuScenes results.
- The data loader only partially handles the raw nuScenes table schema and does
  not implement official splits. An opt-in vehicle-mask generator has synthetic
  tests but no real-data parity evidence. An experimental
  `LSSSegmentation` forward path and BCE/IoU primitives exist, but they are not
  wired into the training/evaluation CLIs or validated against a reference model.
- Existing synthetic tests exercise parts of the scaffold; passing them does not
  demonstrate paper fidelity, checkpoint compatibility, or model performance.

Do not use the `OfficialConfig` name as evidence of the paper's settings: its
current values are not the paper baseline. There are no measured reproduction
results in this repository.

## Development setup

Create a Python environment and install the package and test dependencies:

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

The synthetic tests need no nuScenes dataset or network. The nuScenes dataset,
pretrained weights, and checkpoints are not included. Dependencies and supported
environments still need to be pinned and verified as part of the rebuild.

## Continuous integration

The current GitHub Actions workflow runs the synthetic pytest suite on Python
3.10 and 3.11 and aggregates those results in a `current CI gate` job. This is
an **incomplete** CI foundation: the playbook's lint/format, typing, notebook
execution and parity, package, dependency audit, integration, and documentation
checks still need to be implemented. Branch protection is not yet configured,
so this workflow alone does not enforce those checks before merging.

For commands and their current limitations, see [COMMAND_GUIDE.md](COMMAND_GUIDE.md).
Specs and implementation plans are local-only; public research and verification
evidence will be maintained in the repository.

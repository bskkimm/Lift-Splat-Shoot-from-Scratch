# LSS paper-to-code audit

## Sources and status

- Paper: [Lift, Splat, Shoot](https://arxiv.org/abs/2008.05711), ECCV 2020,
  particularly sections 3–5.
- Reference implementation, inspected but not imported: [nv-tlabs/lift-splat-shoot
  at `2903467c91ee9c12f0917a12c22ab1f04e607ae0`](https://github.com/nv-tlabs/lift-splat-shoot/tree/2903467c91ee9c12f0917a12c22ab1f04e607ae0).
- Reimplementation process: paper-reimplementation playbook revision
  `813ff24b93727ea11c34478fc4c1fc9bbe30ad96`.

**Status: scaffold, not reproduced.** The mapping below distinguishes existing
code from desired behavior. None of the paper metrics, reference checkpoint
parity, or full training have been verified here.

## Paper-to-current-code map

| Paper | Required behavior | Current location and status | Verification still needed |
| --- | --- | --- | --- |
| §3.1, eq. 1; §4.1 | ImageNet-pretrained EfficientNet-B0 image features; predict context and latent depth distribution, then take their outer product. | `lss/models/backbone.py`, `lss/models/depth.py`, `lss/models/lift_splat.py`: simplified image path and depth/context, not the reference encoder. | Encoder initialization, feature shapes, numerical and gradient tests. |
| §3.2; §4.2 | Project frustum points to the ego frame and sum-pool the cameras into BEV pillars. | `lss/geometry.py`, `lss/models/lift_splat.py`: frustum samples use input-image pixel coordinates; fractional bounds and XY cell assignment are tested, but augmentation inversion, output layout and z-bin handling still differ from the reference. | Augmentation geometry, camera-permutation tests, z-bin handling, and reference-layout parity. |
| §4.1 | ResNet-block BEV encoder/decoder to BEV output. | `lss/models/lss.py` returns pooled features and has no paper BEV decoder; `lss/models/head.py` is a detection scaffold. | Segmentation logits and end-to-end tests. |
| §5.2 | Generate vehicle segmentation masks; evaluate BEV intersection-over-union (IoU). | `lss/data/nuscenes_dataset.py` indexes raw camera keyframes and annotation categories, and transforms raw camera/box coordinates to the LiDAR keyframe ego frame. Enriched records still use the legacy global-frame path, missing velocity defaults to zero, and official splits and segmentation masks are absent. `lss/losses.py` and `lss/metrics.py` contain standalone BCE and IoU primitives, not yet connected to training or evaluation. | Real-data frame parity, enriched-record semantics, derived velocity if required, official split, rasterized targets, and validation IoU. |
| §3.3, eq. 2; §5.6 | Score trajectory templates against a cost map and evaluate template rankings. | Not implemented. | Analytical scoring, training and planning-protocol tests. |
| §5 | Train on the paper recipe and report the correct benchmark and split. | `train.py` supports model-construction dry runs only; real training and `eval.py` fail explicitly rather than writing placeholder results. | Real optimizer steps, reproducible resume, data-backed evaluation and measured results. |
| Teaching resource | Standalone architecture definitions and probes consistent with production. | `notebooks/lss_e2e_walkthrough.ipynb` imports the production model. | Independent definitions, fresh-process execution and parity. |

## Paper settings and reference implementation differences

Paper §4.1 describes **128×352** input images, a **200×200** BEV grid over
−50 m to 50 m in each horizontal direction at **0.5 m** resolution, and depths
from 4 m to 45 m at 1 m spacing. Its image encoder uses ImageNet-pretrained
EfficientNet-B0 and its BEV encoder uses ResNet blocks. Paper §5 describes Adam
with learning rate `1e-3`, weight decay `1e-7`, 300k steps and BCE; object
segmentation is reported with positive weight 1.0.

The inspected reference training script uses `dbound=[4.0, 45.0, 1.0]` with
`torch.arange`, so its depth bins exclude 45 m; it sets positive weight 2.13,
batch size 4, and selects five cameras for training by default. These
paper/reference differences must be preserved as explicit recipe choices, not
silently conflated. Pretrained weight provenance and checkpoint compatibility
remain unverified.

The standalone segmentation loss defaults to positive weight 1.0 as in the
paper's object-segmentation description; 2.13 must be supplied explicitly for
the reference training-script setting. IoU thresholds logits at zero; sum
intersection and union counts across validation batches before division. The
empty-union score of 1.0 matches the reference batch metric; full-validation
empty-union handling remains to be defined when that evaluation is implemented.

The reference `get_image_data` passes calibrated camera-to-ego transforms
directly, while its segmentation targets use the LiDAR keyframe's ego pose.
The current raw-data loader additionally compensates for camera-to-LiDAR ego
pose changes. This is a deliberate unvalidated deviation from the reference,
not a demonstrated parity improvement; it needs a named recipe and numerical
comparison before a faithful-configuration claim.

The paper reports BEV segmentation IoU and trajectory-ranking results, not
nuScenes 3D detection mAP/NDS as the LSS baseline. Any detection extension
must be named and evaluated separately. The reference repository reports a
vehicle IoU of 33.03 for its own checkpoint; this is an upstream result, not a
result measured by this repository.

## Rebuild milestones

1. Establish source-grounded documentation, dependency constraints, and CI.
2. Rebuild nuScenes data, camera transformations, and BEV target generation.
3. Implement and verify the faithful lift/splat and BEV segmentation model.
4. Connect real training, evaluation, reproducible checkpoints, and tracking.
5. Add a standalone teaching notebook with package parity checks.
6. Implement and evaluate trajectory scoring and optional map segmentation.
7. Run bounded performance tests and record results with complete provenance.

Full-duration training requires a separately agreed compute budget. Real-data,
GPU, pretrained-checkpoint and official-protocol results remain **not run**
until supported by recorded commands and artifacts.

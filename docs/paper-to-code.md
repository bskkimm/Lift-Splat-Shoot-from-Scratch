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
| §3.1, eq. 1; §4.1 | ImageNet-pretrained EfficientNet-B0 image features; predict context and latent depth distribution, then take their outer product. | `lss/models/segmentation.py` connects the TorchVision camera encoder and latent depth/context lift; its small synthetic forward/backward path is tested. Existing `LSS.forward` remains a separate scaffold. | Pretrained weight provenance, reference checkpoint/numerical parity and real-data testing. |
| §3.2; §4.2 | Project frustum points to the ego frame and sum-pool the cameras into BEV pillars. | `lss/models/segmentation.py` connects image-coordinate geometry and z-aware pooling; fractional bounds, augmentation inversion and small synthetic camera permutation are tested. Existing `LSS.forward` still uses `lift_splat.py`. | Dataset augmentation integration, reference numerical parity and performance verification. |
| §4.1 | ResNet-block BEV encoder/decoder to BEV output. | `LSSSegmentation` uses `lss/models/bev_encoder.py` for raw BEV logits; a small synthetic BCE backward pass is tested. `lss/models/head.py` remains a separate detection scaffold. | Reference checkpoint parity and real training/evaluation tests. |
| §5.2 | Generate vehicle segmentation masks; evaluate BEV intersection-over-union (IoU). | `lss/data/nuscenes_dataset.py` indexes raw camera keyframes and annotation categories, optionally filters official SDK scene splits, transforms raw coordinates to the LiDAR keyframe ego frame, and can emit OpenCV-rasterized vehicle masks. `split=None` is unfiltered. Enriched records still use the legacy global-frame path and missing velocity defaults to zero. BCE and IoU primitives are used by the practical training/validation loop. | Real-data frame, split and raster parity, enriched-record semantics, derived velocity if required, and official validation IoU. |
| §3.3, eq. 2; §5.6 | Score trajectory templates against a cost map and evaluate template rankings. | `lss/models/trajectory.py` implements indexed cost summation, nearest-template selection and classification loss. `lss/models/planner.py` composes a BEV cost-map model with metric-to-grid conversion and template scoring; synthetic backward tests cover the assembled path. It is not wired into a planning dataset or training CLI. | Template generation, masking policy, planning training/evaluation and real-data protocol tests. |
| §5 | Train on the paper recipe and report the correct benchmark and split. | `train.py` connects the segmentation model, named splits, BEV targets, Adam, epoch runner and epoch-boundary checkpointing in a practical FP32 single-process recipe. `eval.py` loads compatible practical checkpoints and reports separately labeled validation IoU. Synthetic CLI tests cover optimization, checkpoint evaluation and interrupted/resumed equivalence with one CPU thread and zero workers. The recipe enables deterministic algorithms and uses direct resize, random initialization, and no paper augmentation/camera dropout. | Reference recipe/weight parity, real-data validation, multiworker/distributed/GPU recovery, official evaluation and measured results. |
| Teaching resource | Standalone architecture definitions and probes consistent with production. | `notebooks/lss_e2e_walkthrough.ipynb` independently defines the practical segmentation model and runs a synthetic optimizer step. Fresh-process tests check model inventory and matched-weight synthetic evaluation and training outputs, gradients, and post-update state. Python socket connections are disabled during smoke execution, not by an OS sandbox. | Full-resolution parity, real-data examples, visualization, pretrained and reference-checkpoint paths, and faithful recipe coverage. |

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

The standalone `CameraEncoder` defaults to random initialization and does not
download weights. Its opt-in `pretrained=True` uses TorchVision's ImageNet-1K
V1 EfficientNet-B0 weights, not the original repository's
`efficientnet_pytorch` weights. Their equivalence, checkpoint key mapping and
impact on LSS output have not been verified. `LSSSegmentation` uses this encoder;
the older `LSS` scaffold does not.

For an offline TorchVision initialization experiment,
`lss.models.initialization.load_torchvision_b0_features` accepts a local
TorchVision EfficientNet-B0 state dictionary, a caller-supplied full SHA-256 and
a descriptive identifier. It verifies the bytes and feature tensor keys,
shapes and dtypes before loading stages 0–7; the unused stage 8, classifier and
LSS-specific fusion/depth layers are not loaded. The caller is responsible for
documenting and verifying the weight source. The training CLI exposes this
helper with an explicit local path, digest and identifier; checkpoint config
records the provenance and resume requires the same verified file. It has
synthetic fixture tests but has not been validated with an official file. It
does not convert an original LSS checkpoint.

The standalone segmentation loss defaults to positive weight 1.0 as in the
paper's object-segmentation description; 2.13 must be supplied explicitly for
the reference training-script setting. IoU thresholds logits at zero; sum
intersection and union counts across validation batches before division. The
empty-union score of 1.0 matches the reference batch metric. The epoch runner
instead reports `iou=None` for an empty aggregate union, because the reference
full-validation ratio is undefined in that case. Training IoU aggregates
predictions measured before each optimizer update and is not an official metric.

The standalone voxel pool uses half-open bounds, floor-based binning and a
z-major `[B,Z*C,X,Y]` output. It rejects grid extents that are not an integral
number of bins, rather than silently truncating the bin count as in the pinned
reference. Its output and gradient parity against a reference checkpoint and
its runtime/memory behavior remain unverified.

The opt-in vehicle target rasterizer uses full three-dimensional annotation
orientation, category prefix `vehicle`, the LiDAR reference ego pose, rounded
grid coordinates and OpenCV polygon filling. Its grid order is `[1,X,Y]`. It
has analytical and synthetic-fixture tests but has not been compared against
reference masks on real nuScenes samples. OpenCV is a data preprocessing
dependency; it is not used by the learnable PyTorch model.
The reference target rasterizer rounds polygon vertices while voxel pooling
assigns points by integer binning; these conventions can differ near cell
boundaries and must not be treated as interchangeable.

When a dataset `split` is explicitly set to `train` or `val`, the loader filters
scene names using the nuScenes devkit's official scene lists, including the
appropriate `mini_train`/`mini_val` names for `v1.0-mini`. The default
`split=None` is an unfiltered diagnostic path and must not be used as a claimed
train/validation reproduction recipe. This selection has synthetic tests only;
the full train-split count and reference parity remain unverified. An opt-in
local validation-loader smoke checked the 6,019-sample split count and one real
sample; its limited scope and resources are recorded in
[`docs/analysis/real-nuscenes-loader-smoke.md`](analysis/real-nuscenes-loader-smoke.md).

The epoch checkpoint helper stores the model, optimizer, optional scheduler,
completed epoch, canonical caller-supplied config, Python/NumPy/PyTorch RNG and
an explicit loader generator; CUDA RNG is opt-in. A synthetic CPU test verifies
an uninterrupted run against an epoch-boundary resume with zero loader workers.
An assembled LSS model also has a synthetic checkpoint save/load output test.
Callers must supply a complete matching config and the generator actually used by
their sampler. Mid-epoch recovery, multiple workers, distributed sampling,
GPU/CUDA restoration and assembled-LSS training continuation remain unverified.
Loading checks common state/optimizer structure before mutation and attempts
rollback after a later failure using CPU snapshots; this requires additional
host memory. Rollback of arbitrary custom state-loading hooks and GPU behavior
has not been verified.

Before deterministic algorithms and a one-thread default were added to the
practical CLI, its synthetic resume comparison failed once on Python 3.10 CI
and passed on an unchanged rerun; the precise cause was not established. The
mitigated configuration has passed repeated CI runs, but determinism across
other hardware, thread counts, precision modes and data-loader workers is not
established.

The practical CLI constructs train and validation datasets independently; each
currently loads complete metadata tables before filtering samples. Full-dataset
memory use and throughput have not been measured.

The reference `get_image_data` passes calibrated camera-to-ego transforms
directly, while its segmentation targets use the LiDAR keyframe's ego pose.
The current raw-data loader additionally compensates for camera-to-LiDAR ego
pose changes. This is a deliberate unvalidated deviation from the reference,
not a demonstrated parity improvement; it needs a named recipe and numerical
comparison before a faithful-configuration claim.

`lss/data/image_augmentation.py` implements the reference-style deterministic
resize/crop/flip/rotate post-transform. The dataset and collator can propagate
explicit per-camera augmentation parameters and post-transforms while keeping
original intrinsics; the practical training recipe does not yet sample these
parameters. An opt-in `ImageAugmentation` dataset policy samples training
transforms from NumPy's RNG and uses a deterministic validation crop. An
opt-in dataset camera subset can sample without replacement during training;
validation retains all cameras. Both are wired into the CLI's separately named
`reference-data` recipe and have synthetic train/resume/evaluation coverage.
Their NumPy RNG behavior is unverified with multiple workers. The augmentation's
continuous flip offset follows the reference convention and is not a claim of
exact PIL pixel-center or real-data parity.

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

# Fourteen worked LSS architecture illustrations

The [standalone walkthrough](../../lss_e2e_walkthrough.ipynb) embeds **14 original
figures** as Markdown PNG attachments. A copied notebook displays the figures
offline without this directory. Code outputs remain empty. Each figure also has
an editable SVG with real text and a compact 1050 × 600 PNG preview here.

These figures explain the **current practical implementation**, not a verified
paper reproduction. Full-size labels use six 128 × 352 validation images, 41
depths `[4,45)` at 1 m spacing, 64 context channels and a 200 × 200 BEV grid.
Reduced values, camera scenes and masks are explicitly illustrative. The
trajectory model is experimental and separate from the segmentation model.

## Inventory and placement

Each image is attached at a stable `fig-<id>` Markdown cell. The adjacent code or
heading is recorded in `manifest.json` and checked automatically. Combined Python
definition cells were split at top-level statement boundaries to place figures
before their corresponding definitions; concatenated Python code is unchanged.

| # | Figure | Exact placement / question answered | Source |
| --- | --- | --- | --- |
| 1 | [pipeline-overview](pipeline-overview.png) | Introduction, before setup. How do cameras become BEV logits and where does planning differ? | `segmentation.py`, `planner.py` |
| 2 | [sample-preparation](sample-preparation.png) | Before §1, introducing the input/target contract before any sample is constructed. Which values feed the model versus supervision? | `nuscenes_dataset.py`, `transforms.py`, `bev_targets.py` |
| 3 | [camera-encoder](camera-encoder.png) | Before `CameraEncoder`. How do 112 and 320 channels fuse into depth and context? | `camera_encoder.py` |
| 4 | [depth-context-lift](depth-context-lift.png) | After the camera probe. What does the outer product do to one pixel's features? | `segmentation.py` |
| 5 | [frustum-geometry](frustum-geometry.png) | Before `make_frustum`. Where does pixel-depth (75,40,4) land in 3-D? | `geometry.py` |
| 6 | [augmentation-frames](augmentation-frames.png) | Before `camera_to_ego`. Which image and ego transforms must be inverted/composed? | `geometry.py`, image augmentation and dataset |
| 7 | [voxel-pooling](voxel-pooling.png) | Before `VoxelPool`. Which points survive and how do features add within height bins? | `voxel_pool.py` |
| 8 | [bev-decoder](bev-decoder.png) | Before `BEVEncoder`. Where do the skip connection and 320-channel concatenation enter? | `bev_encoder.py` |
| 9 | [model-assembly](model-assembly.png) | Before `LSSSegmentation`. Do the value and coordinate streams share the same sample axes? | `segmentation.py`, `voxel_pool.py` |
| 10 | [segmentation-loss](segmentation-loss.png) | Immediately before BCE. How does mean reduction affect each cell's gradient? | `losses.py` |
| 11 | [learning-step](learning-step.png) | Immediately before backward/Adam. Which parameters receive gradients, and what does one scalar update do? | `segmentation_engine.py` |
| 12 | [decoding-iou](decoding-iou.png) | Before inference/visualization. Why are the example intersection and union 2 and 4? | `metrics.py`, segmentation engine |
| 13 | [trajectory-scoring](trajectory-scoring.png) | Before the planning definitions. Why do costs 1 and 5 give probabilities 98.2% and 1.8%? | `trajectory.py`, `planner.py` |
| 14 | [checkpoint-evaluation](checkpoint-evaluation.png) | Before the opt-in paths. How does saved configuration control real evaluation? | `eval.py`, checkpoint and data recipes |

Full source paths and digests are in the manifest. Each SVG has the same basename
as its PNG. Learning, decoding and metric operations share one figure only where
their worked example stays readable; BCE and the optimizer update are separate.

## Regenerate and verify

From the repository root, using an environment with the project's dependencies
(Matplotlib and Pillow are also required for rendering):

```bash
python scripts/illustrate_notebook.py --render
python scripts/illustrate_notebook.py --attach
python scripts/illustrate_notebook.py --check
MPLBACKEND=Agg python scripts/execute_notebook.py notebooks/lss_e2e_walkthrough.ipynb
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest -q \
  tests/test_notebook_illustrations.py tests/test_notebook_architecture.py
```

`--render` writes only PNG/SVG and the manifest; `--attach` inserts inventoried
Markdown cells and checks that the notebook's Python AST is unchanged. Repeated
attachment is byte-identical. `--check` is read-only: it validates source and
generator digests, image/attachment hashes, exact coverage/placement, captions,
unique cell IDs and the existing 500 KiB notebook size limit. It fails on stale,
missing, corrupt, duplicate and orphaned inventoried assets.

For a style contact sheet, choose an output path outside this source directory:

```bash
python scripts/illustrate_notebook.py --contact-sheet /tmp/lss-architecture.png
```

The drawings use `scripts/architecture/`: a shared vector canvas plus camera,
geometry, decoder and learning lessons. Arithmetic comes from `examples.py`;
tests compare it with analytical literals and the package's geometry, pooling,
BCE, Adam, IoU and trajectory operators. The renderer rejects text outside the
canvas, overflowing card labels and overlapping card title/body text. Every
figure also needs visual inspection; those geometric checks do not certify
scientific correctness or catch all possible overlaps.

## Provenance and rendering environment

- Visualization standard adopted from playbook revision
  `707c64b138d5e6c534af3d197fd4f278aab03d45`.
- Instructional inspiration: PlanTF at
  `2e02a5c485ef2edd88555610143828a7a122c96f`. No PlanTF images or drawing code are
  copied. All LSS figures, including the overview, are programmatic vector art;
  no image-generation tool or external image asset was used.
- Recorded renderer: Python 3.11, Matplotlib 3.11.0, Pillow 12.0.0, NumPy 1.26.4,
  FreeType 2.14.3, DejaVu Sans. The manifest includes the exact font digest.
- SVG IDs are fixed, timestamps omitted and inputs deterministic. Exact bytes
  depend on rendering-library/font versions. Tests render twice in their current
  environment; comparison to retained asset bytes additionally applies when
  the recorded rendering environment matches. Reading existing PNG attachments
  does not require this exact environment.

The pre-commit size limit is preserved. Raster previews use an optimized palette;
SVGs retain vector detail for zooming. These diagrams do not provide new evidence
of full-resolution reference parity, official checkpoint compatibility, trained
vehicle IoU or real-data planning quality.

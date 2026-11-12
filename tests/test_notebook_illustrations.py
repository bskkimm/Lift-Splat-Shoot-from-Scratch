"""Catch missing, stale or nonportable illustrations in the teaching notebook."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "notebooks/assets/architecture"
NOTEBOOK = ROOT / "notebooks/lss_e2e_walkthrough.ipynb"
FIGURES = {
    "pipeline-overview", "sample-preparation", "camera-encoder", "depth-context-lift",
    "frustum-geometry", "augmentation-frames", "voxel-pooling", "bev-decoder",
    "model-assembly", "segmentation-loss", "learning-step", "decoding-iou",
    "trajectory-scoring", "checkpoint-evaluation",
}


def run_tool(*args):
    env = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run([sys.executable, str(ROOT / "scripts/illustrate_notebook.py"), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)


def test_all_inventoried_figures_are_portable_markdown_attachments():
    result = run_tool("--check")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    assert {figure["id"] for figure in manifest["figures"]} == FIGURES
    notebook = json.loads(NOTEBOOK.read_text())
    attached = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        for name in re.findall(r"!\[(?:\\.|[^\]\\])*\]\(attachment:([^)]+)\)", source):
            data = base64.b64decode(cell["attachments"][name]["image/png"], validate=True)
            assert data == (ASSETS / name).read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            attached.append(name.removesuffix(".png"))
    assert len(attached) == len(FIGURES) and set(attached) == FIGURES
    assert NOTEBOOK.stat().st_size < 500 * 1024


def test_checker_rejects_corrupted_asset(tmp_path):
    assets = tmp_path / "architecture"
    shutil.copytree(ASSETS, assets)
    (assets / "voxel-pooling.png").write_bytes(b"broken image")
    result = run_tool("--check", "--assets", str(assets))
    assert result.returncode != 0
    assert "voxel-pooling" in result.stderr and "hash" in result.stderr.lower()


def test_checker_cannot_silently_drop_source_freshness_checks(tmp_path):
    assets = tmp_path / "architecture"
    shutil.copytree(ASSETS, assets)
    manifest_path = assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["figures"][0]["sources"] = {}
    manifest_path.write_text(json.dumps(manifest))
    result = run_tool("--check", "--assets", str(assets))
    assert result.returncode != 0
    assert "source coverage" in result.stderr


def test_checker_rejects_corrupted_embedded_image(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "fig-camera-encoder")
    cell["attachments"]["camera-encoder.png"]["image/png"] = base64.b64encode(b"wrong bytes").decode()
    path = tmp_path / "copied.ipynb"
    path.write_text(json.dumps(notebook))
    result = run_tool("--check", "--notebook", str(path))
    assert result.returncode != 0
    assert "camera-encoder: notebook attachment hash mismatch" in result.stderr


def test_reattachment_is_idempotent_and_preserves_notebook_code(tmp_path):
    path = tmp_path / "copy.ipynb"
    shutil.copyfile(NOTEBOOK, path)
    before = path.read_bytes()
    result = run_tool("--attach", "--notebook", str(path))
    assert result.returncode == 0, result.stderr
    assert path.read_bytes() == before


def test_generation_is_repeatable_in_the_same_rendering_environment(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for destination in (first, second):
        result = run_tool("--render", "--assets", str(destination))
        assert result.returncode == 0, result.stderr
    for path in first.iterdir():
        if path.suffix in {".png", ".svg", ".json"}:
            other = second / path.name
            assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(other.read_bytes()).digest(), path.name
    recorded = json.loads((ASSETS / "manifest.json").read_text())
    regenerated = json.loads((first / "manifest.json").read_text())
    if recorded["environment"] == regenerated["environment"]:
        for figure in recorded["figures"]:
            for suffix in ("png", "svg"):
                assert (first / figure[suffix]).read_bytes() == (ASSETS / figure[suffix]).read_bytes()


def test_geometry_and_pooling_examples_match_package_computation():
    import torch
    from lss.geometry import camera_to_ego
    from lss.models.voxel_pool import VoxelPool

    data = json.loads((ASSETS / "manifest.json").read_text())["worked_examples"]
    geometry = data["geometry"]
    point = torch.tensor(geometry["pixel_depth"]).reshape(1, 1, 1, 1, 1, 3)
    intrinsic = torch.tensor(geometry["intrinsics"]).reshape(1, 1, 3, 3)
    transform = torch.tensor([[0., -1., 0., 1.], [1., 0., 0., 2.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.]]).reshape(1, 1, 4, 4)
    actual = camera_to_ego(point, intrinsic, transform)
    torch.testing.assert_close(actual.flatten(), torch.tensor([1., 3., 4.]))
    assert geometry["camera_xyz"] == [1., 0., 4.] and geometry["ego_xyz"] == [1., 3., 4.]
    augmented = torch.tensor([80., 8., 1.]).reshape(1, 1, 1, 1, 1, 3)
    post_rot = torch.diag(torch.tensor([-.5, .5, 1.])).reshape(1, 1, 3, 3)
    post_trans = torch.tensor([105., -2., 0.]).reshape(1, 1, 3)
    raw = camera_to_ego(augmented, torch.eye(3)[None, None], torch.eye(4)[None, None], post_rot, post_trans)
    torch.testing.assert_close(raw.flatten(), torch.tensor([50., 20., 1.]))

    example = data["pooling"]
    points = torch.tensor(example["points"]).reshape(1, 1, 1, 1, 6, 3)
    features = torch.tensor(example["features"]).reshape(1, 1, 1, 1, 6, 2).requires_grad_()
    pooled = VoxelPool((0, 2, 1), (0, 2, 1), (0, 2, 1))(points, features)
    torch.testing.assert_close(pooled, torch.tensor(example["z_c_x_y"]).reshape(1, 4, 2, 2))
    pooled.sum().backward()
    torch.testing.assert_close(features.grad.reshape(6, 2), torch.tensor([[1., 1.], [1., 1.], [1., 1.],
                                                                       [0., 0.], [0., 0.], [1., 1.]]))
    assert data["lift"]["outer_product"] == [[.5, -.25, 1.], [1.5, -.75, 3.]]


def test_loss_update_metric_and_trajectory_examples_match_package():
    import torch
    from lss.losses import binary_segmentation_loss
    from lss.metrics import segmentation_intersection_union
    from lss.models.trajectory import trajectory_logits

    data = json.loads((ASSETS / "manifest.json").read_text())["worked_examples"]
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    loss = binary_segmentation_loss(logits, torch.tensor(data["bce"]["target"])[None, None])
    loss.backward()
    torch.testing.assert_close(loss.detach(), torch.tensor(data["bce"]["mean_loss"]))
    torch.testing.assert_close(logits.grad[0, 0], torch.tensor(data["bce"]["mean_logit_gradient"]))
    z = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=.1)
    torch.nn.functional.binary_cross_entropy_with_logits(z, torch.ones_like(z)).backward()
    optimizer.step()
    torch.testing.assert_close(z.detach(), torch.tensor([data["adam"]["updated_logit"]], dtype=z.dtype))
    iou = data["iou"]
    intersection, union = segmentation_intersection_union(torch.tensor(iou["logits"])[None, None],
                                                         torch.tensor(iou["target"])[None, None])
    assert (int(intersection), int(union)) == (2, 4)
    assert iou["iou"] == .5
    trajectory = data["trajectory"]
    scores = trajectory_logits(torch.tensor(trajectory["cost_map"])[None],
                               torch.tensor(trajectory["cells"])[None])
    torch.testing.assert_close(scores[0], torch.tensor([-1., -5.]))
    torch.testing.assert_close(scores.softmax(dim=1)[0], torch.tensor(trajectory["probabilities"]))

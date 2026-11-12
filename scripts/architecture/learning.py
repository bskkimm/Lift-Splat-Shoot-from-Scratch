"""Supervise and decode: losses, gradients, IoU and trajectory probabilities."""

import numpy as np
from matplotlib.patches import Rectangle

from .canvas import Canvas, color


def segmentation_loss(data):
    d = data["bce"]
    c = Canvas(10, "Supervise the BEV logits directly", "Binary cross entropy couples each vehicle-occupancy label to one raw output logit.")
    c.panel(34, 148, 370, 541, "A  /  Binary labels", "Toy 2 × 2 grid; positive weight = 1", "gold")
    c.matrix(129, 282, d["target"], (86, 72), "gold", rows=["x0", "x1"], cols=["y0", "y1"])
    c.text(219, 507, "y ∈ {0,1}", 25, "gold", True, "center")
    c.text(219, 578, "All four cells are supervised.", 16, ha="center")
    c.panel(434, 148, 502, 541, "B  /  Stable logit-space objective", "Worked case: z=0 gives probability 0.5", "teal")
    c.text(685, 285, "ℓ(z,y) = softplus(z) − y·z", 23, "teal", True, "center")
    c.text(685, 352, "ℓ(0,y) = log(2) ≈ 0.6931", 22, ha="center")
    c.text(685, 421, "L = mean over B, K, X and Y", 19, ha="center")
    c.card(476, 502, 418, 132, "BCEWithLogits(logits, target)", "No sigmoid before BCE\nmean loss ≈ 0.6931", "teal", 17)
    c.panel(966, 148, 400, 541, "C  /  Logit gradients", "∂L/∂z = (sigmoid(z) − y) / 4", "red")
    c.matrix(1045, 282, d["mean_logit_gradient"], (117, 72), "red", highlight=[(0, 0), (1, 1)])
    c.text(1166, 508, "negative → increase logit", 17, "red", ha="center")
    c.text(1166, 558, "positive → decrease logit", 17, "red", ha="center")
    c.text(1166, 615, "Gradients include mean reduction.", 15, ha="center")
    c.note("Notebook uses positive weight 1.0; the pinned reference training script uses 2.13. This diagram reports no trained loss.")
    return c


def learning_step(data):
    d = data["adam"]
    c = Canvas(11, "Backward reaches the entire feature path", "Red arrows show derivatives; geometric cell membership is a discrete routing decision.")
    stages = [(34, "image encoder", "blue"), (304, "depth × context", "purple"),
              (574, "voxel sum", "teal"), (844, "BEV decoder", "teal"), (1114, "BCE loss", "gold")]
    for x, name, tone in stages:
        c.card(x, 173, 252, 86, name, tone=tone, size=18)
    for x in (286, 556, 826, 1096):
        c.arrow((x, 210), (x + 17, 210), tone="gray")
        c.arrow((x + 17, 303), (x, 303), tone="red")
    c.path([(1240, 259), (1240, 303), (160, 303), (160, 259)], "red", width=2.3)
    c.text(700, 339, "gradients to weights, depth logits and context — not through floor(cell indices)", 17, "red", True, "center")
    c.panel(34, 392, 402, 297, "A  /  One illustrative scalar", "Single positive target; logit z=0", "gold")
    c.text(235, 505, "L = softplus(−z)", 22, "gold", True, "center")
    c.text(235, 565, "g = sigmoid(0) − 1 = −0.5", 19, ha="center")
    c.text(235, 628, "L before ≈ 0.6931", 20, ha="center")
    c.panel(466, 392, 432, 297, "B  /  First Adam update", "β₁=0.9; β₂=0.999; ε=10⁻⁸; lr=0.1", "red")
    c.text(682, 505, f"m₁ = {d['first_moment']:.2f}     v₁ = {d['second_moment']:.5f}", 20, "red", True, "center")
    c.text(682, 565, "m̂₁ = −0.5          v̂₁ = 0.25", 20, ha="center")
    c.text(682, 628, "z₁ = z₀ − lr · m̂₁ / (√v̂₁ + ε)", 18, ha="center")
    c.panel(928, 392, 438, 297, "C  /  Inspect the change", "The actual notebook update follows below", "teal")
    c.text(1147, 507, f"z: 0 → {d['updated_logit']:.1f}", 27, "teal", True, "center")
    c.text(1147, 571, f"L after ≈ {d['updated_loss']:.4f}", 22, "teal", True, "center")
    c.text(1147, 630, "Toy learning rate differs from lr=10⁻³ below.", 14, ha="center")
    c.note("Scalar arithmetic illustrates Adam; it is not a measured learning curve or an assertion that every model step reduces loss.")
    return c


def decoding(data):
    d = data["iou"]
    logits = np.array(d["logits"])
    truth = np.array(d["target"], dtype=bool)
    prediction = logits > 0
    c = Canvas(12, "Decode occupancy, then count overlap", "A probability threshold of 0.5 is the same as logit > 0; aggregate counts before dividing.")
    panels = [(34, "A  /  Predicted occupancy", prediction.astype(int), "blue"),
              (486, "B  /  Ground-truth vehicles", truth.astype(int), "gold"),
              (938, "C  /  Intersection", (prediction & truth).astype(int), "teal")]
    for x, title, grid, tone in panels:
        c.panel(x, 154, 428, 322, title, "Illustrative grid: rows=x; columns=y", tone)
        c.matrix(x + 65, 293, grid, (100, 62), tone, highlight=[(0, 0), (1, 1)])
    c.card(34, 521, 428, 154, "Inference mode", "model.eval() + no_grad()\nraw logit → sigmoid for display\nBatchNorm uses running statistics", "blue", 19)
    c.card(486, 521, 428, 154, "Union = predicted OR target", "predicted=3; target=3\nunion = 3 + 3 − 2 = 4", "gold", 18)
    c.card(938, 521, 428, 154, f"IoU = {d['intersection']} / {d['union']} = {d['iou']:.2f}",
           "Sum counts across batches\nEmpty union → undefined", "teal", 23)
    c.note("Toy masks explain IoU; they are not nuScenes predictions. Practical validation IoU is not a reproduced paper score.")
    return c


def trajectory(data):
    d = data["trajectory"]
    c = Canvas(13, "Shoot: compare paths by accumulated cost", "The planning primitive predicts a separate cost map; every template point must lie inside the grid.", "EXPERIMENTAL PLANNING")
    c.panel(34, 148, 414, 541, "A  /  One toy cost map", "Rows=x; columns=y; no interpolation", "purple")
    c.matrix(144, 301, d["cost_map"], (100, 95), "purple", rows=["x0", "x1"], cols=["y0", "y1"])
    c.arrow((194, 345), (294, 345), tone="blue")
    c.arrow((194, 442), (294, 442), tone="red")
    c.text(241, 544, "A: (0,0) → (0,1)", 19, "blue", True, "center")
    c.text(241, 602, "B: (1,0) → (1,1)", 19, "red", True, "center")
    c.panel(478, 148, 410, 541, "B  /  Negative summed cost", "logit[k] = −Σₜ cost[cell(k,t)]", "teal")
    c.card(507, 288, 352, 107, "Candidate A", "−(0 + 1) = −1", "blue", 19)
    c.card(507, 429, 352, 107, "Candidate B", "−(2 + 3) = −5", "red", 19)
    c.text(683, 614, "Repeated cells contribute repeatedly.", 15, ha="center")
    c.panel(918, 148, 448, 541, "C  /  Normalize over templates", "p(k) = softmax(logits)[k]", "purple")
    for i, (p, tone, label) in enumerate(zip(d["probabilities"], ("blue", "red"), ("A", "B"))):
        y = 305 + i * 135
        c.text(959, y + 20, label, 20, tone, True)
        c.ax.add_patch(Rectangle((996, y), 282, 43, facecolor=color("pale_gray"), edgecolor="none"))
        c.ax.add_patch(Rectangle((996, y), 282 * p, 43, facecolor=color(tone), edgecolor="none"))
        c.text(996, y + 72, f"p({label}) ≈ {100 * p:.1f}%", 19, tone, True)
    c.text(1142, 616, "Lower accumulated cost is preferred.", 16, ha="center")
    c.note("Schematic costs and paths. No planning dataset, full training or validated planning metric is implied.")
    return c


def checkpoint(data):
    c = Canvas(14, "Take real paths only when requested", "Checkpoint configuration controls model/data construction; missing requested assets must fail clearly.", "OPT-IN INFRASTRUCTURE")
    c.card(34, 165, 394, 130, "Local nuScenes assets", "Explicit dataroot + version\nofficial validation split", "blue", 20)
    c.card(34, 398, 394, 168, "Practical checkpoint", "format_version=1; config + model\noptimizer/RNG for training resume", "purple", 19)
    c.arrow((428, 229), (501, 229), tone="blue")
    c.card(501, 165, 377, 130, "Dataset + preprocessing", "images + K/T + targets\nsaved preprocessing recipe", "blue", 18)
    c.arrow((428, 481), (501, 481), tone="purple")
    c.card(501, 398, 377, 168, "Configured evaluation model", "Build matching shapes/bounds\nstrictly load local weights\neval() — no optimizer update", "purple", 18)
    c.arrow((689, 398), (689, 295), "config", "purple")
    c.arrow((878, 229), (953, 229), tone="blue")
    c.arrow((878, 481), (953, 481), tone="purple")
    c.panel(953, 165, 413, 401, "Evaluate the package path", "eval.py uses the saved dataset version", "teal")
    c.text(1159, 310, "Forward all selected validation batches", 16, ha="center")
    c.arrow((1159, 340), (1159, 390))
    c.text(1159, 429, "Sum intersection + union", 20, "teal", True, "center")
    c.text(1159, 495, "practical_bev_vehicle_iou", 18, "teal", ha="center")
    c.card(34, 605, 1332, 88, "Independent opt-ins: one real sample · local checkpoint · package parity · full practical evaluation",
           "The default executes only synthetic CPU examples. No original-checkpoint compatibility or paper metric is claimed.", "gold", 17)
    c.note("The notebook checkpoint-load demo and package evaluation are distinct operations; this is an orchestration illustration.")
    return c

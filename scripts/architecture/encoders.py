"""Observe and lift: original visual explanations for camera features."""

from matplotlib.patches import Rectangle

from .canvas import Canvas, color


def overview(data):
    c = Canvas(1, "Lift, Splat, Shoot", "From six calibrated views to one ego-frame grid — follow the colored evidence.")
    for x, title, body, tone in [(34, "01  OBSERVE", "Shared image features", "blue"),
                                  (375, "02  LIFT", "A distribution along each ray", "purple"),
                                  (716, "03  SPLAT", "Sum evidence in metric cells", "teal"),
                                  (1057, "04  PREDICT", "Decode a top-down field", "gold")]:
        c.panel(x, 142, 309, 400, title, body, tone)
    for i, label in enumerate(("front left", "front", "front right", "back left", "back", "back right")):
        c.camera(58 + (i % 2) * 136, 235 + (i // 2) * 86, 112, 58, label)
    c.arrow((190, 510), (423, 510), "shared encoder", "blue")
    c.stack(411, 237, 110, 78, "purple", label="context c")
    c.text(612, 235, "toy D=3", 12, "purple", ha="center")
    for i, (height, value) in enumerate(((28, ".10"), (63, ".25"), (112, ".65"))):
        x = 566 + i * 33
        c.ax.add_patch(Rectangle((x, 371 - height), 23, height, facecolor=color("purple"), edgecolor="white"))
        c.text(x + 12, 392, value, 12, "purple", ha="center")
    c.text(540, 430, "p(depth) × context", 17, "purple", ha="center")
    c.text(540, 469, "[N, 41, 8, 22, 64]", 14, ha="center")
    c.plane(750, 454, 168, 99)
    for offset, tone in ((0, "blue"), (50, "purple"), (100, "teal")):
        apex = (770 + offset, 250)
        for endpoint in ((805, 415), (920, 425), (956, 356)):
            c.path([apex, endpoint], tone, width=1.1)
        c.ax.scatter(*apex, color=color(tone), s=35, zorder=5)
    c.text(868, 491, "[64, 200, 200]", 15, "teal", ha="center")
    c.arrow((683, 355), (738, 355), tone="purple")
    c.arrow((1025, 355), (1074, 355))
    values = [[-2, -2, -2, -2], [-2, 2, 2, -2], [-2, 2, -2, -2], [-2, -2, -2, -2]]
    c.matrix(1105, 248, values, (52, 42), "gold", highlight=((1, 1), (1, 2), (2, 1)))
    c.text(1212, 457, "vehicle logits", 18, "gold", bold=True, ha="center")
    c.text(1212, 493, "[1, 200, 200]", 15, ha="center")
    c.card(34, 575, 810, 116, "Primary path: BEV vehicle segmentation",
           "BCE supervision → gradients through decoder,\npooled features and image encoder", "teal", 18)
    c.card(884, 575, 482, 116, "Separate cost-map model",
           "same architecture; trajectory costs\nnot shared segmentation weights", "purple", 17)
    c.note("Practical full-size example: B=1, N=6; 41 depths (4 ≤ d < 45 m). Spatial artwork is schematic.")
    return c


def sample(data):
    c = Canvas(2, "Prepare views, geometry and supervision", "The camera branch and annotation branch meet in the same reference ego frame.")
    c.panel(34, 142, 408, 548, "A  /  RGB views", "Six named cameras; one shared encoder", "blue")
    for i in range(6):
        c.camera(60 + (i % 2) * 182, 246 + (i // 2) * 112, 146, 76)
    c.text(238, 620, "[N, 3, 128, 352]", 19, "blue", True, "center")
    c.panel(474, 142, 414, 548, "B  /  Consistent geometry", "Choose a named preprocessing recipe", "teal")
    c.card(500, 236, 362, 106, "Practical: direct resize", "Resize also scales K", "blue")
    c.card(500, 375, 362, 128, "Reference-data recipe",
           "Resize/crop/flip/rotate\nkeep K; record post-transform", "purple", 17)
    c.text(681, 552, "RGB in [0,1] → (RGB − mean) / std", 16, ha="center")
    c.text(681, 597, "R channel: (0.5 − 0.485) / 0.229", 15, "muted", ha="center")
    c.text(681, 634, "= 0.0655   (illustrative pixel)", 17, "teal", True, "center")
    c.panel(920, 142, 446, 548, "C  /  Vehicle occupancy target", "Global 3-D boxes → LiDAR-reference ego", "gold")
    c.matrix(1016, 275, [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
             (68, 56), "gold", rows=["x0", "x1", "x2", "x3"], cols=["y0", "y1", "y2", "y3"])
    c.text(1143, 541, "bottom polygons → round → fill", 16, "gold", ha="center")
    c.text(1143, 590, "target [1, 200, 200] ∈ {0,1}", 17, ha="center")
    c.text(1143, 632, "Not an input to the forward pass", 15, "red", ha="center")
    c.note("Data preparation is infrastructure; the standalone default below uses a seeded, explicitly synthetic sample.")
    return c


def camera_encoder(data):
    c = Canvas(3, "One encoder, two spatial scales", "Flatten B × N only for image processing; every camera uses the same weights.")
    c.card(34, 164, 244, 100, "Images", "[BN, 3, 128, 352]", "blue")
    c.arrow((278, 213), (350, 213), tone="blue")
    c.card(350, 164, 292, 100, "EfficientNet 0–5", "stride 16; [BN,112,8,22]", "blue", 17)
    c.arrow((642, 213), (721, 213), tone="blue")
    c.card(721, 164, 292, 100, "Stages 6–7", "stride 32; [BN,320,4,11]", "purple")
    c.arrow((1013, 213), (1090, 213), tone="purple")
    c.card(1090, 164, 276, 100, "Bilinear upsample", "[BN,320,8,22]", "purple", 17)
    c.stack(381, 338, 129, 92, "blue", label="112-channel skip")
    c.stack(763, 338, 129, 92, "purple", label="320-channel feature")
    c.arrow((498, 264), (449, 330), "skip", "blue")
    c.path([(1226, 264), (1226, 390), (930, 390)], "purple")
    c.arrow((929, 390), (914, 390), tone="purple")
    c.arrow((526, 387), (604, 387), tone="blue")
    c.arrow((756, 387), (708, 387), tone="purple")
    c.card(584, 344, 144, 88, "CONCAT", "112 + 320", "teal", 17)
    c.arrow((656, 432), (656, 515), "432 channels")
    c.card(390, 515, 532, 94, "Two 3×3 Conv + BN + ReLU blocks", "432 → 512 → 512; preserve 8 × 22", "teal", 18)
    c.arrow((922, 562), (972, 562), "1×1", "teal")
    c.path([(974, 527), (974, 641)], "teal")
    c.arrow((974, 527), (1000, 527))
    c.arrow((974, 641), (1000, 641))
    c.card(1000, 482, 366, 90, "Depth logits: 41", "softmax over depth → p(d)", "purple")
    c.card(1000, 596, 366, 90, "Context features: 64", "no softmax over context", "blue")
    c.text(34, 651, "Projection: 512 → 105 = 41 + 64", 18, "teal", True)
    c.note("Full-size shapes shown. Random initialization by default; opt-in TorchVision weights are not official LSS weights.")
    return c


def lift(data):
    d = data["lift"]
    c = Canvas(4, "Lift: multiply a distribution by a feature", "One pixel becomes several depth-weighted feature vectors, not one chosen depth.")
    c.panel(34, 150, 306, 540, "A  /  Depth probabilities", "Toy D=2; probabilities sum to 1", "purple")
    for i, p in enumerate(d["depth"]):
        x = 104 + 106 * i
        c.ax.add_patch(Rectangle((x, 480 - p * 280), 60, p * 280,
                                facecolor=color("purple"), edgecolor="white"))
        c.text(x + 30, 515, f"d{i}", 16, "purple", ha="center")
        c.text(x + 30, 480 - p * 280 - 24, f"{p:.2f}", 19, "purple", True, "center")
    c.text(187, 598, "p(d) = softmax(depth logits)", 15, ha="center")
    c.panel(372, 150, 365, 540, "B  /  Context at that pixel", "Toy C=3; values may be negative", "blue")
    c.matrix(432, 326, [d["context"]], (80, 68), "blue", cols=["c0", "c1", "c2"])
    c.text(555, 477, "c = [2, −1, 4]", 22, "blue", True, "center")
    c.text(555, 560, "No direct depth labels are needed.", 15, ha="center", fit_width=320)
    c.text(555, 596, "Segmentation supervises the lift.", 15, ha="center", fit_width=320)
    c.text(755, 367, "⊗", 32, "teal", ha="center")
    c.panel(786, 150, 580, 540, "C  /  Lifted outer product", "Each row retains the entire context vector", "teal")
    c.matrix(934, 307, d["outer_product"], (112, 72), "teal", rows=["d0 × c", "d1 × c"],
             cols=["c0", "c1", "c2"], highlight=[(0, 0), (1, 0)])
    c.text(1076, 520, "0.25 × 2 = 0.5      0.75 × 2 = 1.5", 17, "teal", True, "center")
    c.text(1076, 580, "[B,N,D,h,w,C]", 23, "teal", True, "center")
    c.text(1076, 622, "Full example: [1,6,41,8,22,64]", 17, ha="center")
    c.note("Illustrative values. Gradients reach both the softmax depth probabilities and the context channels.")
    return c

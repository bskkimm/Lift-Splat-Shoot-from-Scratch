"""Observe and lift: spatially illustrated camera and feature computation."""

import numpy as np
from matplotlib.patches import Circle, Rectangle

from .canvas import Canvas, color
from .illustration import (frustum_volume, ground_grid, occupancy,
                           road_image, surround_scene, texture, volume)


def overview(data):
    c = Canvas(1, "Lift, Splat, Shoot — an illustrated pipeline",
               "Follow one visual signal from surrounding cameras through 3-D evidence into a top-down prediction.")
    for x, title, tone in ((34, "01 / OBSERVE", "blue"), (374, "02 / LIFT", "purple"),
                           (720, "03 / SPLAT", "teal"), (1030, "04 / PREDICT", "gold")):
        c.text(x, 164, title, 18, tone, True)
    surround_scene(c, 164, 330, 86)
    c.text(164, 451, "ego + six camera directions", 13, "blue", ha="center")
    for i in range(6):
        road_image(c, 42 + (i % 3) * 89, 481 + (i // 3) * 66, 74, 47, i)
    c.text(164, 623, "six RGB views · shared encoder", 14, "blue", ha="center")
    c.arrow((301, 350), (357, 350), tone="blue")

    frustum_volume(c, (390, 405), (204, -65), width=124, height=184, layers=4)
    c.text(568, 239, "categorical depth planes", 14, "purple", ha="center")
    volume(c, 384, 510, 92, 65, 21, "blue", texture(4, 6), (2, 3))
    for i, p in enumerate(data["lift"]["depth"]):
        c.ax.add_patch(Rectangle((536 + i * 52, 581 - p * 100), 32, p * 100,
                                 facecolor=color("purple"), edgecolor="white", zorder=4))
        c.text(552 + i * 52, 600, f"{p:g}", 12, "purple", ha="center")
    c.text(580, 481, "toy p(d)", 12, "purple", ha="center")
    c.text(452, 604, "context c", 13, "blue", ha="center")
    c.text(529, 649, "p(d) ⊗ c → [N,D,h,w,C]", 16, "purple", True, "center")
    c.arrow((669, 350), (715, 350), tone="purple")

    point = ground_grid(c, 728, 485, 142, 100, texture(5, 5, .8), pillars=45)
    target = point(.46, .5, 27)
    for i, tone in enumerate(("blue", "purple", "teal")):
        origin = (748 + i * 68, 278 + 12 * (i % 2))
        c.ax.add_patch(Circle(origin, 5, facecolor=color(tone), edgecolor="white", zorder=7))
        c.arrow(origin, target, tone=tone)
    c.ax.add_patch(Circle(target, 6, facecolor=color("red"), edgecolor="white", zorder=8))
    c.text(830, 553, "same cell → sum", 17, "teal", True, "center")
    c.text(830, 593, "[64,200,200]", 15, "teal", ha="center")
    c.arrow((952, 355), (1008, 355))

    for x, y, w, h, d in ((1030, 300, 48, 108, 18), (1112, 329, 35, 72, 24),
                            (1190, 304, 54, 106, 18)):
        volume(c, x, y, w, h, d, "teal", texture(6, 4))
    c.arrow((1098, 351), (1110, 351))
    c.arrow((1173, 351), (1188, 351))
    c.text(1164, 246, "BEV decoder", 17, "teal", True, "center")
    ground_grid(c, 1091, 555, 173, 83, occupancy(), "gold")
    c.arrow((1230, 421), (1230, 470), tone="gold")
    c.text(1203, 597, "vehicle logit field", 16, "gold", True, "center")
    c.text(1203, 632, "[1,200,200]", 14, "gold", ha="center")
    c.path([(35, 695), (1365, 695)], "line", width=.8)
    c.text(34, 710, "Separate experimental model: same architecture → cost map → trajectory-template scores", 14, "purple")
    c.note("Original schematic art and toy feature fields; full practical dimensions: six 128×352 views, D=41, C=64.")
    return c


def sample(data):
    c = Canvas(2, "Build a consistent multi-camera sample",
               "Visual observations and vehicle annotations take separate routes into a common reference ego frame.")
    c.text(34, 170, "OBSERVATIONS", 18, "blue", True)
    for i, name in enumerate(("front", "front right", "back right", "back", "back left", "front left")):
        road_image(c, 40 + (i % 3) * 176, 235 + (i // 3) * 141, 147, 100, i, name)
    surround_scene(c, 282, 607, 56)
    c.text(282, 700, "[N,3,128,352]", 16, "blue", True, "center")

    c.text(614, 170, "PIXELS + CALIBRATION", 18, "teal", True)
    road_image(c, 627, 253, 153, 97, 2)
    c.ax.add_patch(Rectangle((690, 301), 20, 17, facecolor="none", edgecolor=color("red"), lw=2, zorder=9))
    c.arrow((700, 358), (700, 412), tone="teal")
    volume(c, 635, 438, 125, 86, 24, "blue", texture(4, 6))
    c.text(715, 562, "resize also scales K", 15, "teal", ha="center")
    c.text(715, 606, "augmentation instead records", 13, "muted", ha="center")
    c.text(715, 636, "post-rotation + translation", 14, "purple", ha="center")
    c.text(715, 688, "R: (0.5 − .485)/.229 = .0655", 14, "teal", ha="center")

    c.text(965, 170, "SUPERVISION ONLY", 18, "gold", True)
    values = occupancy()
    ground_grid(c, 978, 426, 210, 142, values, "gold", pillars=27)
    c.text(1162, 473, "vehicle box footprints", 17, "gold", True, "center")
    c.arrow((1162, 498), (1162, 540), tone="gold")
    ground_grid(c, 1015, 664, 166, 83, (values > 0).astype(float), "gold")
    c.text(1162, 699, "binary target [1,200,200]", 16, "gold", True, "center")
    c.note("Scenes and footprints are illustrative. Real annotations use global 3-D boxes and the LiDAR-reference ego pose.")
    return c


def camera_encoder(data):
    c = Canvas(3, "The camera encoder as a feature pyramid",
               "Image sites contract while channel depth grows; a fine-scale skip recovers spatial detail.")
    road_image(c, 34, 269, 154, 111, 0)
    c.text(111, 415, "RGB image", 16, "blue", True, "center")
    c.text(111, 454, "128 × 352", 14, "blue", ha="center")
    c.arrow((192, 327), (247, 327), tone="blue")
    volume(c, 253, 263, 79, 134, 17, "blue", texture(8, 5))
    volume(c, 410, 288, 73, 106, 30, "blue", texture(7, 5, .4), (3, 3))
    volume(c, 592, 324, 48, 70, 48, "purple", texture(5, 4, .8))
    volume(c, 756, 288, 73, 106, 48, "purple", texture(7, 5, .8))
    for start, end in (((350, 340), (405, 340)), ((520, 350), (586, 350)), ((690, 354), (750, 354))):
        c.arrow(start, end, tone="purple")
    for x, name, shape in ((294, "early stages", ""), (454, "stage-5 skip", "112 × 8 × 22"),
                            (642, "stages 6–7", "320 × 4 × 11"), (811, "upsample", "320 × 8 × 22")):
        c.text(x, 443, name, 14, "blue" if x < 500 else "purple", True, "center")
        if shape:
            c.text(x, 481, shape, 13, ha="center")
    c.path([(453, 398), (453, 565), (922, 565), (922, 358)], "blue", width=2.4)
    c.text(688, 596, "112-channel long skip", 16, "blue", True, "center")
    c.ax.add_patch(Circle((920, 350), 16, facecolor="white", edgecolor=color("teal"), lw=1.8, zorder=7))
    c.text(920, 350, "∥", 17, "teal", True, "center", zorder=8)
    c.arrow((879, 350), (901, 350), tone="purple")
    c.arrow((938, 350), (965, 350))
    volume(c, 971, 286, 72, 120, 47, "teal", texture(7, 5))
    c.text(1031, 462, "fusion", 16, "teal", True, "center")
    c.text(1031, 501, "432 → 512 → 512", 13, ha="center")
    c.arrow((1095, 345), (1147, 345), "1×1")
    c.path([(1149, 278), (1149, 523)], "teal")
    volume(c, 1202, 233, 98, 75, 20, "purple", texture(4, 6))
    volume(c, 1202, 487, 98, 75, 30, "blue", texture(4, 6))
    c.arrow((1150, 271), (1198, 271), tone="purple")
    c.arrow((1150, 523), (1198, 523), tone="blue")
    c.text(1253, 354, "41 depth bins", 16, "purple", True, "center")
    c.text(1253, 390, "depth softmax", 13, "purple", ha="center")
    c.text(1253, 611, "64 context channels", 15, "blue", True, "center")
    c.text(676, 673, "∥ = channel concatenation    ·    textured cuboids = illustrative feature tensors", 16, "teal", ha="center")
    c.note("Full practical dimensions, excluding BN/ReLU details shown in code. Feature textures and cuboid thickness are schematic.")
    return c


def lift(data):
    d = data["lift"]
    c = Canvas(4, "Lift the same image evidence along a ray",
               "A feature vector is distributed across depth planes; the latent depth probabilities control its strength.")
    road_image(c, 43, 236, 278, 180, 1)
    c.ax.add_patch(Circle((229, 349), 8, facecolor="none", edgecolor=color("red"), lw=2.4, zorder=9))
    c.text(182, 466, "one selected image site", 17, "blue", True, "center")
    c.arrow((229, 421), (229, 510), tone="red")
    volume(c, 100, 556, 160, 70, 24, "blue", np.array([[.5, .1, 1.]]))
    c.text(194, 670, "context c = [2,−1,4]", 17, "blue", True, "center")

    c.text(489, 213, "toy D=2", 16, "purple", True, "center")
    for i, p in enumerate(d["depth"]):
        x = 414 + i * 103
        c.ax.add_patch(Rectangle((x, 464 - p * 220), 60, p * 220,
                                 facecolor=color("purple"), edgecolor="white", zorder=4))
        c.text(x + 30, 505, f"p(d{i})\n{p:g}", 14, "purple", True, "center")
    c.text(490, 580, "p(d) ⊗ c", 27, "purple", True, "center")
    c.arrow((604, 354), (701, 354), tone="purple")

    frustum_volume(c, (735, 380), (365, -50), width=167, height=205, layers=2)
    c.text(1096, 207, "depth-weighted 3-D evidence", 18, "purple", True, "center")
    c.text(1110, 462, "retain every context channel", 16, "teal", True, "center")
    c.matrix(958, 546, d["outer_product"], (104, 54), "teal",
             rows=["d0 × c", "d1 × c"], cols=["c0", "c1", "c2"], highlight=[(0, 0), (1, 0)])
    c.text(1100, 692, "0.25 × 2 = 0.5;  0.75 × 2 = 1.5", 16, "teal", True, "center")
    c.note("Illustrative textures; the numerical table is exact. Full lifted shape: [B,N,D,h,w,C]; depth needs no direct labels.")
    return c

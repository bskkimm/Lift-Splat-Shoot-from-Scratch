"""Calibrate and splat: ray geometry, image transforms and voxel arithmetic."""

import numpy as np
from matplotlib.patches import Polygon

from .canvas import Canvas, color


def frustum(data):
    d = data["geometry"]
    c = Canvas(5, "A pixel becomes a calibrated 3-D ray", "Depth is optical-axis z in metres; image coordinates are pixels of the original input grid.")
    c.panel(34, 148, 394, 544, "A  /  Sample the frustum", "8 × 22 feature sites; 41 depth bins", "purple")
    origin = np.array([92, 468])
    for i, scale in enumerate((.55, .78, 1.)):
        center = origin + np.array([260, -160]) * scale
        size = np.array([115, 140]) * scale
        polygon = [center + [-size[0] / 2, -size[1] / 2], center + [size[0] / 2, -size[1] / 2],
                   center + [size[0] / 2, size[1] / 2], center + [-size[0] / 2, size[1] / 2]]
        c.ax.add_patch(Polygon(polygon, facecolor=color("pale_purple"), edgecolor=color("purple"), alpha=.65))
        if i == 2:
            for corner in polygon:
                c.path([origin, corner], "purple", width=1)
    c.ax.scatter(*origin, s=65, color=color("blue"), zorder=5)
    c.text(83, 507, "camera", 14, "blue")
    c.text(220, 578, "u: 0 … 351     v: 0 … 127", 16, ha="center")
    c.text(220, 624, "d: 4, 5, …, 44 m (45 excluded)", 15, "purple", ha="center")
    c.panel(458, 148, 408, 544, "B  /  Undo intrinsics", "Hand-check one illustrative point", "blue")
    c.matrix(503, 277, d["intrinsics"], (100, 52), "blue")
    c.text(662, 471, "(u,v,d) = (75,40,4)", 19, "blue", True, "center")
    c.text(662, 524, "K⁻¹ [u·d, v·d, d]ᵀ", 22, ha="center")
    c.text(662, 585, "camera XYZ = (1,0,4) m", 20, "blue", True, "center")
    c.arrow((430, 420), (457, 420), tone="purple")
    c.panel(896, 148, 470, 544, "C  /  Rotate, then translate", "p_ego = R · p_camera + t", "teal")
    c.arrow((1000, 427), (1235, 427), "ego x", "teal")
    c.arrow((1000, 427), (1000, 250), "", "teal")
    c.text(971, 241, "ego y", 14, "teal")
    c.path([(1000, 427), (1044, 295)], "gray", True)
    c.ax.scatter([1044], [295], s=110, color=color("teal"), zorder=5)
    c.text(1069, 279, "(1,3)", 17, "teal", True)
    c.text(1131, 504, "Toy R = 90° about z; t = (1,2,0)", 16, ha="center")
    c.text(1131, 566, "ego XYZ = (1,3,4) m", 21, "teal", True, "center")
    c.text(1131, 625, "[B,N,D,h,w,3]", 20, ha="center")
    c.note("Ray planes are schematic. Column-vector equations match camera_to_ego; toy K/R/t are not nuScenes calibrations.")
    return c


def augmentation(data):
    c = Canvas(6, "Undo pixels; compose the correct ego frame", "Image augmentation and sensor motion are different transforms, applied in a precise order.")
    c.panel(34, 148, 1332, 261, "A  /  A worked continuous image transform", "Original uv=(50,20) → resized/cropped/flipped uv=(80,8)", "purple")
    cards = [(64, "original", "(50,20)", "blue"), (365, "resize × 0.5", "(25,10)", "purple"),
             (666, "crop − (5,2)", "(20,8)", "purple"), (967, "flip: 100 − u", "(80,8)", "purple")]
    for x, title, value, tone in cards:
        c.card(x, 264, 265, 95, title, value, tone, 17)
    for x in (329, 630, 931):
        c.arrow((x, 310), (x + 33, 310), tone="purple")
    c.text(700, 385, "Unprojection reverses this: post_rot⁻¹ (pixel_depth − post_trans), then K⁻¹", 17, "teal", True, "center")
    c.panel(34, 440, 1332, 268, "B  /  The repository's timestamp-compensated camera transform", "Reference targets use the LiDAR keyframe ego pose", "teal")
    for x, title, body in [(64, "camera", "sensor coordinates"), (398, "camera-time ego", "calibrated sensor"),
                           (732, "global", "camera-time pose"), (1066, "LiDAR-time ego", "inverse ref. pose")]:
        c.card(x, 536, 270, 106, title, body, "teal", 17)
    for x in (334, 668, 1002):
        c.arrow((x, 588), (x + 62, 588))
    c.text(700, 676, "R = Rrefᵀ Rpose Rcal       t = Rrefᵀ (Rpose tcal + tpose − tref)", 19, "teal", True, "center")
    c.note("Continuous flip convention; no claim of exact PIL pixel-center parity. Pinned original LSS uses calibration directly.")
    return c


def pooling(data):
    d = data["pooling"]
    c = Canvas(7, "Splat is a masked sum, not an average", "Follow points A and B into the same cell; rejected points contribute neither values nor gradients.")
    c.panel(34, 148, 426, 545, "A  /  XY slice at z = 0.2 m", "Toy bounds: 0 ≤ x,y,z < 2; step=1 m", "teal")
    for i in range(3):
        c.path([(115 + i * 135, 319), (115 + i * 135, 589)], "gray", width=.8)
        c.path([(115, 319 + i * 135), (385, 319 + i * 135)], "gray", width=.8)
    for x, y, label, tone in [(142, 562, "A: [2,4]", "blue"), (223, 548, "B: [3,1]", "purple"),
                               (277, 427, "C: [7,2]", "teal")]:
        c.ax.scatter(x, y, s=75, color=color(tone), zorder=5)
        c.text(x - 20, y - 29, label, 14, tone)
    c.ax.scatter([101, 385], [562, 562], marker="x", s=70, color=color("red"), zorder=6)
    c.text(78, 634, "x = −0.1 rejected; x = 2.0 rejected", 15, "red")
    c.text(99, 282, "y (m)", 14, "muted")
    c.text(343, 609, "x (m)", 14, "muted")
    c.text(250, 664, "The upper bound is excluded.", 15, "red", ha="center")
    c.panel(490, 148, 386, 545, "B  /  Keep → floor → sum", "index = floor((point − lower) / step)", "blue")
    c.matrix(538, 286, [["A", "(0,0,0)", "yes"], ["B", "(0,0,0)", "yes"],
                       ["C", "(1,1,0)", "yes"], ["D/E", "outside", "no"]],
             (96, 51), "blue", cols=["point", "cell XYZ", "keep"], disabled=[(3, 0), (3, 1), (3, 2)])
    c.card(518, 539, 330, 107, "Cell (0,0,0)", "[2,4] + [3,1] = [5,5]", "teal", 18)
    c.panel(906, 148, 460, 545, "C  /  Height × channels", "Output [B,Z·C,X,Y]; toy Z=2, C=2", "purple")
    c.text(1136, 244, "tables: rows=x, columns=y", 13, "muted", ha="center")
    for i, (name, grid) in enumerate(zip(("z0 · c0", "z0 · c1", "z1 · c0", "z1 · c1"),
                                       np.array(d["z_c_x_y"]).reshape(4, 2, 2))):
        x = 949 + (i % 2) * 200; y = 310 + (i // 2) * 180
        c.text(x + 65, y - 35, name, 16, "purple", True, "center")
        c.matrix(x, y, grid, (67, 49), "purple", highlight=[(0, 0)])
    c.text(1136, 657, "A z=1.2 point contributes [5,6].", 15, ha="center")
    c.note("Worked feature channels are illustrative. Floor indices are discrete; gradients pass through selected lifted features.")
    return c

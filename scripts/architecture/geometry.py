"""Spatial ray, frame and voxel illustrations with exact numerical examples."""

import numpy as np
from matplotlib.patches import Circle

from .canvas import Canvas, color
from .illustration import (callout, face, frustum_volume, ground_grid, road_image,
                           vehicle)


def frustum(data):
    c = Canvas(5, "See the frustum, not just the equation",
               "An image site traces one camera ray; categorical depth samples locate feature evidence along it.")
    road_image(c, 42, 238, 245, 159, 1)
    c.ax.add_patch(Circle((225, 317), 8, facecolor="none", edgecolor=color("red"), lw=2.1, zorder=9))
    c.text(164, 441, "toy (u,v,d) = (75,40,4)", 16, "blue", True, "center")
    c.arrow((300, 318), (389, 375), tone="red")
    frustum_volume(c, (410, 417), (336, -54), 154, 249, layers=6)
    c.text(686, 193, "image/depth planes", 18, "purple", True, "center")
    c.text(686, 527, "41 bins: 4,5,…,44 m", 17, "purple", True, "center")
    c.text(686, 565, "selected ray shown in red", 14, "red", ha="center")
    c.text(164, 516, "K (toy calibration)", 16, "blue", True, "center")
    c.matrix(60, 560, data["geometry"]["intrinsics"], (70, 39), "blue")
    c.text(615, 635, "K⁻¹[ud,vd,d]ᵀ = (1,0,4) m", 21, "blue", True, "center")
    c.text(615, 685, "optical-axis depth, not Euclidean ray length", 13, "muted", ha="center")

    c.text(1134, 193, "ROTATE + TRANSLATE", 18, "teal", True, "center")
    origin = np.array([1145., 465.])
    basis = np.array([[39., 12.], [-25., 13.], [0., -35.]])
    for axis, name in zip(basis, ("x", "y", "z")):
        end = origin + axis * 3
        c.arrow(origin, end)
        c.text(*(end + [0, -17]), name, 14, "teal", ha="center")
    point = origin + np.array([1., 3., 4.]) @ basis
    c.path([origin, origin + basis[0], origin + basis[0] + 3 * basis[1], point], "gray", True)
    c.ax.add_patch(Circle(point, 7, facecolor=color("teal"), edgecolor="white", zorder=8))
    c.text(*(point + [-15, -20]), "(1,3,4)", 17, "teal", True, ha="right")
    vehicle(c, 1246, 480, .68, angle=-35, tone="teal")
    c.text(1140, 566, "R: 90° about z; t=(1,2,0)", 16, "teal", ha="center")
    c.text(1140, 631, "ego XYZ = (1,3,4) m", 20, "teal", True, "center")
    c.text(1140, 682, "[B,N,D,h,w,3]", 17, ha="center")
    c.note("A subset of planes is drawn. Toy intrinsics/poses are independent of the schematic street and full-size image grid.")
    return c


def augmentation(data):
    c = Canvas(6, "Keep image transforms and ego motion distinct",
               "The image coordinate is restored before unprojection; the sensor point is then moved between ego frames.")
    for i, (x, title, uv) in enumerate(((41, "original", "(50,20)"), (376, "resize × 0.5", "(25,10)"),
                                       (711, "crop − (5,2)", "(20,8)"), (1046, "flip: 100 − u", "(80,8)"))):
        crop = (.025, .025, .525, .775) if i >= 2 else None
        road_image(c, x, 226, 248, 142, 0, mirror=i == 3, crop=crop)
        px, py = ((.125, .125), (.125, .125), (.2, 8 / 60), (.8, 8 / 60))[i]
        c.ax.add_patch(Circle((x + 248 * px, 226 + 142 * py), 7, facecolor="none", edgecolor=color("red"), lw=2, zorder=9))
        c.text(x + 124, 181, title, 17, "blue" if i == 0 else "purple", True, "center")
        c.text(x + 124, 398, uv, 19, "purple", True, "center")
        c.text(x + 124, 435, ("400×160 px", "200×80 px", "100×60 px", "100×60 px")[i], 11, "muted", ha="center")
        if i < 3:
            c.arrow((x + 253, 291), (x + 320, 291), tone="purple")
    c.text(700, 475, "undo post-transform → intrinsic unprojection → camera-to-reference-ego transform", 17, "teal", True, "center")
    # The lower story is a distinct illustration of sensor motion, not a pixel operation.
    ground_grid(c, 157, 672, 213, 90, np.zeros((5, 5)), "gray")
    vehicle(c, 255, 621, .54, -18, "blue")
    vehicle(c, 368, 590, .54, -6, "teal")
    c.path([(259, 618), (310, 602), (368, 590)], "purple", True, 1.5)
    c.text(260, 700, "camera-time ego", 14, "blue", ha="center")
    c.text(396, 528, "LiDAR-time ego", 14, "teal", ha="center")
    c.arrow((445, 613), (523, 613))
    c.text(839, 550, "camera → camera-time ego → global → reference ego", 17, "teal", True, "center")
    c.text(839, 613, "R = Rrefᵀ Rpose Rcal", 22, "teal", True, "center")
    c.text(839, 673, "t = Rrefᵀ (Rpose tcal + tpose − tref)", 20, "teal", True, "center")
    c.note("Schematic image/frame illustrations; uv arithmetic follows the continuous post-transform convention, not PIL pixel centers.")
    return c


def pooling(data):
    d = data["pooling"]
    c = Canvas(7, "Watch features accumulate inside voxel pillars",
               "In-bounds points are routed to discrete cells; all their context channels are summed without averaging.")
    origin = np.array([103., 574.])

    def project(p):
        x, y, z = p
        return origin + [114 * x + 51 * y, -64 * y - 118 * z]

    # A transparent 2×2×2 wire volume with cell boundaries and true toy point coordinates.
    for z in (0, 1, 2):
        for i in range(3):
            for line in (((i, 0, z), (i, 2, z)), ((0, i, z), (2, i, z))):
                points = [project(p) for p in line]
                c.path(points, "line", width=.8)
    for x in range(3):
        for y in range(3):
            c.path([project((x, y, 0)), project((x, y, 2))], "line", width=.65)
    for end, name, offset in (((2.25, -.2, 0), "x", (4, 14)),
                              ((-.2, 2.3, 0), "y", (-14, -3)),
                              ((-.2, -.2, 2.3), "z", (-11, -12))):
        base = project((-.2, -.2, 0))
        tip = project(end)
        c.arrow(base, tip, tone="gray")
        c.text(*(tip + offset), name, 13, "muted", ha="center")
    bottom = [project(p) for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
    top = [p - [0, 118] for p in bottom]
    face(c, [bottom[0], bottom[1], top[1], top[0]], "teal", .72, alpha=.45, zorder=2)
    face(c, top, "teal", .8, alpha=.45, zorder=2)
    for i, (p, valid) in enumerate(zip(d["points"], d["valid"])):
        pt = project(p)
        tone = ("blue", "purple", "teal", "red", "red", "gold")[i]
        c.ax.scatter(*pt, marker="o" if valid else "x", s=48, color=color(tone), zorder=8)
        offset = (-19, 6) if i == 3 else ((-12, -20) if i in (0, 1) else (12, -20))
        c.text(pt[0] + offset[0], pt[1] + offset[1], chr(65 + i), 14, tone, True,
               ha="right" if i in (0, 1, 3) else "left")
        if i in (0, 1):
            c.path([pt, project((.5, .5, .45))], tone, True)
    callout(c, 48, 178, 1, "Actual toy XYZ points", "blue", 15)
    c.text(260, 648, "0 ≤ x,y,z < 2 m; Δ=1 m", 16, "teal", True, "center")
    c.text(260, 689, "D/E rejected; F is in height bin z1", 14, "muted", ha="center")
    c.arrow((460, 427), (539, 427))

    callout(c, 555, 178, 2, "Channel-wise sum", "teal", 15)
    c.matrix(586, 303, [[2, 4], [3, 1]], (74, 61), "blue", rows=["A", "B"], cols=["c0", "c1"])
    c.text(660, 468, "+", 30, "teal", True, "center")
    c.matrix(586, 516, [[5, 5]], (74, 61), "teal", cols=["c0", "c1"], highlight=[(0, 0), (0, 1)])
    c.text(660, 636, "[2,4] + [3,1] = [5,5]", 17, "teal", True, "center")
    c.text(660, 685, "sum, not mean", 14, "teal", ha="center")
    c.arrow((777, 427), (851, 427), tone="purple")

    callout(c, 872, 178, 3, "Height-major channel layout", "purple", 15)
    for i, (label, values) in enumerate(zip(("z0 · c0", "z0 · c1", "z1 · c0", "z1 · c1"),
                                          np.array(d["z_c_x_y"]).reshape(4, 2, 2))):
        x = 874 + (i % 2) * 241; y = 395 + (i // 2) * 216
        point = ground_grid(c, x, y, 125, 67, values / 7, "purple")
        c.text(x + 80, y - 113, label, 17, "purple", True, "center")
        for u in range(2):
            for v in range(2):
                center = point((u + .5) / 2, (v + .5) / 2)
                c.text(*center, f"{values[u, v]:g}", 15, "ink", True, "center", zorder=7)
    c.text(1108, 691, "[B,Z·C,X,Y] — preserve both z and C", 16, "purple", True, "center")
    c.note("Toy point positions and sums are exact. Wireframe is a parallel projection; gradients flow to features, not floor indices.")
    return c

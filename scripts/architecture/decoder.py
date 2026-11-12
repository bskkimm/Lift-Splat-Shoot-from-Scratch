"""Illustrated multiscale BEV decoding and dual-stream model assembly."""

from matplotlib.patches import Circle

from .canvas import Canvas, color
from .illustration import (frustum_volume, ground_grid, occupancy, road_image,
                           surround_scene, texture, volume)


def bev_decoder(data):
    c = Canvas(8, "A spatial view of the BEV encoder–decoder",
               "Read surface size as spatial resolution, extrusion as channels, and the blue route as the fine-scale skip.")
    ground_grid(c, 40, 407, 146, 95, texture(6, 6), "teal")
    c.text(144, 466, "input BEV", 17, "teal", True, "center")
    c.text(144, 504, "64 × 200 × 200", 14, ha="center")
    c.arrow((254, 344), (315, 344))

    for x, y, w, h, d, tone in ((324, 284, 82, 136, 24, "blue"),
                                (524, 325, 66, 95, 34, "purple"),
                                (714, 361, 45, 59, 48, "purple"),
                                (937, 285, 82, 136, 48, "purple"),
                                (1171, 285, 82, 136, 40, "teal")):
        volume(c, x, y, w, h, d, tone, texture(6, 5, x / 300))
    for x, title, dims, tone in ((371, "stem + layer1", "64 × 100 × 100", "blue"),
                                 (571, "layer2", "128 × 50 × 50", "purple"),
                                 (766, "layer3", "256 × 25 × 25", "purple"),
                                 (1000, "bilinear upsample", "256 × 100 × 100", "purple"),
                                 (1227, "fusion", "320 → 256 → 256", "teal")):
        c.text(x, 217, title, 16, tone, True, "center")
        c.text(x, 466, dims, 13, tone, ha="center")
    for start, end in (((432, 358), (518, 358)), ((627, 382), (708, 382)),
                        ((810, 384), (928, 351))):
        c.arrow(start, end, tone="purple")

    c.path([(366, 423), (366, 563), (1117, 563), (1117, 355)], "blue", width=2.5)
    c.text(730, 603, "64-channel spatial skip", 19, "blue", True, "center")
    c.ax.add_patch(Circle((1117, 346), 17, facecolor="white", edgecolor=color("teal"), lw=1.7, zorder=7))
    c.text(1117, 346, "∥", 17, "teal", True, "center", zorder=8)
    c.arrow((1071, 346), (1097, 346), tone="purple")
    c.arrow((1137, 346), (1167, 346))

    c.arrow((1240, 429), (1240, 519))
    ground_grid(c, 1151, 651, 135, 75, occupancy(), "gold")
    c.text(1228, 697, "raw logits [B,1,200,200]", 16, "gold", True, "center")
    c.text(35, 660, "upsample to 200² → 3×3 Conv 128 → BN/ReLU → 1×1 Conv", 17, "teal", True)
    c.note("Full practical dimensions. No stem max-pool and no sigmoid in the model. Feature textures/voxel surfaces are illustrative.")
    return c


def assembly(data):
    c = Canvas(9, "Two streams meet at the same 3-D sample sites",
               "Lifted values say what is present; calibrated coordinates say where each contribution belongs.")
    c.text(35, 171, "VALUES / image evidence", 18, "blue", True)
    road_image(c, 45, 240, 157, 105, 0)
    road_image(c, 65, 260, 157, 105, 2)
    c.text(143, 402, "[1,6,3,128,352]", 14, "blue", True, "center")
    c.arrow((230, 309), (320, 309), tone="blue")
    volume(c, 330, 237, 107, 97, 23, "blue", texture(6, 7))
    volume(c, 494, 264, 80, 72, 37, "purple", texture(5, 6))
    c.text(469, 386, "depth × context", 19, "purple", True, "center")
    c.text(469, 426, "[1,6,41,8,22] × [1,6,64,8,22]", 13, ha="center")
    c.arrow((618, 309), (737, 309), tone="purple")
    volume(c, 746, 241, 118, 113, 67, "purple", texture(8, 8))
    c.text(840, 398, "lifted values", 17, "purple", True, "center")
    c.text(840, 438, "[1,6,41,8,22,64]", 14, "purple", ha="center")

    c.text(35, 483, "COORDINATES / calibration", 18, "teal", True)
    surround_scene(c, 158, 584, 59)
    c.text(158, 695, "K, T, image post-transform", 14, "teal", ha="center")
    c.arrow((265, 574), (373, 574))
    frustum_volume(c, (392, 586), (349, -22), 148, 147, 4, "teal")
    c.text(637, 696, "ego XYZ [1,6,41,8,22,3]", 16, "teal", True, "center")

    c.path([(932, 317), (985, 317), (985, 461)], "purple", width=2)
    c.arrow((826, 554), (966, 481), tone="teal")
    c.ax.add_patch(Circle((986, 484), 24, facecolor="white", edgecolor=color("teal"), lw=2, zorder=8))
    c.text(986, 484, "Σ", 23, "teal", True, "center", zorder=9)
    c.text(985, 542, "voxel sum", 16, "teal", True, "center")

    ground_grid(c, 1080, 383, 165, 88, texture(6, 6), "teal")
    c.arrow((1011, 472), (1127, 405))
    c.text(1192, 426, "[1,64,200,200]", 14, "teal", True, "center")
    c.arrow((1190, 447), (1190, 481))
    volume(c, 1152, 488, 63, 68, 22, "teal", texture(5, 5))
    c.text(1278, 523, "BEV\ndecoder", 14, "teal", True, "center")
    c.arrow((1190, 565), (1190, 599), tone="gold")
    ground_grid(c, 1124, 678, 132, 57, occupancy(), "gold")
    c.text(1220, 710, "[1,1,200,200] logits", 15, "gold", True, "center")
    c.note("Practical B=1, six validation cameras; D=41, C=64 and Z=1. Shape labels are exact; depicted feature patterns are schematic.")
    return c

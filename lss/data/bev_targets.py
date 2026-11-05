"""Vehicle box projection and reference-style BEV mask rasterization."""

import math

import cv2
import numpy as np
import torch

from lss.geometry import quaternion_to_matrix


def vehicle_bottom_polygon(annotation, reference_rotation, reference_translation):
    """Return four box bottom corners in the reference ego frame, in metres.

    nuScenes boxes use size [width, length, height], global translation, and
    quaternion orientation [w,x,y,z]. The reference rotation maps ego to
    global coordinates and the translation is the ego origin in global metres.
    """
    width, length, height = annotation["size"]
    corners = reference_rotation.new_tensor([
        [length / 2, -width / 2, -height / 2],
        [length / 2, width / 2, -height / 2],
        [-length / 2, width / 2, -height / 2],
        [-length / 2, -width / 2, -height / 2],
    ])
    box_rotation = quaternion_to_matrix(reference_rotation.new_tensor(annotation["rotation"]))
    center = reference_rotation.new_tensor(annotation["translation"])
    corners_global = corners @ box_rotation.T + center
    return ((corners_global - reference_translation) @ reference_rotation)[:, :2]


def rasterize_polygons(polygons, x_bounds, y_bounds):
    """Rasterize ego XY polygons as a float [1,X,Y] occupancy mask.

    Grid coordinates are rounded before OpenCV polygon filling, as in the
    reference target generator. OpenCV receives (column=y, row=x) vertices.
    """
    sizes = []
    for lower, upper, step in (x_bounds, y_bounds):
        if not all(math.isfinite(value) for value in (lower, upper, step)) or step <= 0 or upper <= lower:
            raise ValueError("BEV bounds must be finite and increasing with positive step")
        count = (upper - lower) / step
        if round(count) < 1 or not math.isclose(count, round(count), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("BEV bounds must define an integral number of bins")
        sizes.append(int(round(count)))
    mask = np.zeros(tuple(sizes), dtype=np.uint8)
    lower = np.asarray([x_bounds[0], y_bounds[0]])
    step = np.asarray([x_bounds[2], y_bounds[2]])
    for polygon in polygons:
        xy = torch.as_tensor(polygon).detach().cpu().numpy()
        if xy.shape != (4, 2) or not np.isfinite(xy).all():
            raise ValueError("each polygon must have four finite XY vertices")
        indices = np.round((xy - lower) / step).astype(np.int32)
        cv2.fillPoly(mask, [indices[:, [1, 0]]], 1)
    return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

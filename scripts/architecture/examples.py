"""Small computed examples displayed in the figures (not model measurements)."""

import numpy as np


def worked_examples():
    depth = np.array([0.25, 0.75])
    context = np.array([2.0, -1.0, 4.0])
    intrinsic = np.array([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]])
    pixel_depth = np.array([75., 40., 4.])
    camera = np.linalg.solve(intrinsic, np.r_[pixel_depth[:2] * pixel_depth[2], pixel_depth[2]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    ego = rotation @ camera + [1., 2., 0.]
    points = np.array([[.2, .2, .2], [.8, .3, .2], [1.2, 1.2, .2],
                       [-.1, .2, .2], [2., .2, .2], [.2, .2, 1.2]])
    features = np.array([[2., 4.], [3., 1.], [7., 2.], [99., 99.], [99., 99.], [5., 6.]])
    valid = ((points >= 0) & (points < 2)).all(axis=1)
    pooled = np.zeros((2, 2, 2, 2))  # z, c, x, y
    for point, feature in zip(points[valid], features[valid]):
        x, y, z = np.floor(point).astype(int)
        pooled[z, :, x, y] += feature
    labels = np.array([[1., 0.], [0., 1.]])
    loss = np.logaddexp(0, np.zeros_like(labels))
    gradient = (.5 - labels) / labels.size
    z, grad = 0., -.5
    moment, variance = .1 * grad, .001 * grad**2
    updated = z - .1 * (moment / .1) / (np.sqrt(variance / .001) + 1e-8)
    logits = np.array([[2., -1., .2], [-2., 3., -.5]])
    truth = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    prediction = logits > 0
    intersection = int((prediction & truth).sum())
    union = int((prediction | truth).sum())
    costs = np.array([[0., 1.], [2., 3.]])
    cells = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
    scores = -costs[cells[..., 0], cells[..., 1]].sum(axis=1)
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    return {
        "lift": {"depth": depth.tolist(), "context": context.tolist(),
                 "outer_product": np.outer(depth, context).tolist()},
        "geometry": {"intrinsics": intrinsic.tolist(), "pixel_depth": pixel_depth.tolist(),
                     "camera_xyz": camera.tolist(), "ego_xyz": ego.tolist()},
        "augmentation": {"original_uv": [50., 20.], "resize": .5, "crop_xy": [5., 2.],
                         "flip_width": 100., "augmented_uv": [80., 8.]},
        "pooling": {"points": points.tolist(), "features": features.tolist(),
                    "valid": valid.tolist(), "z_c_x_y": pooled.tolist()},
        "bce": {"target": labels.tolist(), "mean_loss": float(loss.mean()),
                "mean_logit_gradient": gradient.tolist()},
        "adam": {"initial_logit": z, "gradient": grad, "learning_rate": .1,
                 "first_moment": moment, "second_moment": variance,
                 "updated_logit": float(updated), "updated_loss": float(np.logaddexp(0, -updated))},
        "iou": {"logits": logits.tolist(), "target": truth.astype(int).tolist(),
                "intersection": intersection, "union": union, "iou": intersection / union},
        "trajectory": {"cost_map": costs.tolist(), "cells": cells.tolist(),
                       "logits": scores.tolist(), "probabilities": probabilities.tolist()},
    }

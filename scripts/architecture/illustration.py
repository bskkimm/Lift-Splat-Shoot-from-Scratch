"""Spatial vector artwork for the architecture lessons.

Vehicles, camera views and feature textures are original schematic artwork.
They are not photographs, dataset examples or measured network activations.
"""

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.transforms import Affine2D

from .canvas import color


def tint(tone, amount=.5):
    rgb = np.array(to_rgb(color(tone)))
    return tuple(rgb * (1 - amount) + amount)


def face(c, points, tone, light=.65, alpha=1, zorder=2, edge=True):
    patch = Polygon(points, closed=True, facecolor=tint(tone, light),
                    edgecolor=color(tone) if edge else "none", lw=.8,
                    alpha=alpha, zorder=zorder, joinstyle="round")
    c.ax.add_patch(patch)
    return patch


def vehicle(c, x, y, scale=1, angle=0, tone="blue"):
    """A roof, glazing, wheels and body in a rotatable top-down vehicle icon."""
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def polygon(points, fill, edge="none", order=5):
        points = np.array(points) * scale @ rotation.T + [x, y]
        patch = Polygon(points, facecolor=fill, edgecolor=edge, lw=.8, zorder=order)
        c.ax.add_patch(patch)

    polygon([[-24, -36], [24, -36], [24, 41], [-24, 41]], tint("gray", .86), order=3)
    for side in (-1, 1):
        for offset in (-23, 22):
            polygon([[side * 18 - 4, offset - 9], [side * 18 + 4, offset - 9],
                     [side * 18 + 4, offset + 9], [side * 18 - 4, offset + 9]], color("ink"), order=4)
    polygon([[-17, -41], [17, -41], [21, -27], [21, 30], [16, 41], [-16, 41], [-21, 30], [-21, -27]],
            tint(tone, .22), color(tone))
    polygon([[-13, -25], [13, -25], [15, -10], [-15, -10]], color("ink"))
    polygon([[-15, 13], [15, 13], [12, 27], [-12, 27]], color("ink"))
    polygon([[-13, -8], [13, -8], [13, 11], [-13, 11]], tint(tone, .68), color(tone))
    for side in (-1, 1):
        polygon([[side * 12 - 3, -36], [side * 12 + 3, -36], [side * 12 + 3, -31], [side * 12 - 3, -31]], "white")


def road_image(c, x, y, w, h, variant=0, label=None, mirror=False, crop=None):
    """A miniature street illustration with perspective, buildings and traffic."""
    original_artists = set(c.ax.get_children())
    c.ax.add_patch(Rectangle((x, y), w, h, facecolor="#EAF1F7", edgecolor=color("blue"), lw=1, zorder=2))
    horizon = y + .39 * h
    vanishing = x + w * (.5 + .06 * np.sin(variant))
    # A small skyline and verges establish perspective rather than an empty box.
    for i, frac in enumerate((.04, .14, .24, .71, .83, .93)):
        bh = h * (.15 + .065 * ((i + variant) % 3))
        c.ax.add_patch(Rectangle((x + frac * w, horizon - bh), .085 * w, bh,
                                 facecolor=tint("gray", .58 + .08 * (i % 2)), edgecolor="none", zorder=3))
    face(c, [(x, horizon), (x + w, horizon), (x + w, y + h), (x, y + h)], "teal", .91, edge=False, zorder=3)
    face(c, [(vanishing - .07 * w, horizon), (vanishing + .07 * w, horizon),
             (x + .98 * w, y + h), (x + .02 * w, y + h)], "gray", .54, edge=False, zorder=4)
    for fraction in (.25, .75):
        top = vanishing + (fraction - .5) * .14 * w
        bottom = x + fraction * w
        for start, stop in ((.08, .2), (.3, .48), (.65, .93)):
            c.ax.plot([top + (bottom - top) * start, top + (bottom - top) * stop],
                      [horizon + (y + h - horizon) * start, horizon + (y + h - horizon) * stop],
                      color="white", lw=.75, zorder=5)
    for fraction, depth, tone in ((.64, .7, "blue"), (.36, .38, "red")):
        cy = horizon + (y + h - horizon) * depth
        cx = vanishing + (x + fraction * w - vanishing) * depth
        cw, ch = w * .15 * depth, h * .25 * depth
        face(c, [(cx - cw / 2, cy), (cx - cw * .4, cy - ch), (cx + cw * .4, cy - ch),
                 (cx + cw / 2, cy)], tone, .16, zorder=6)
        face(c, [(cx - cw * .29, cy - ch * .77), (cx + cw * .29, cy - ch * .77),
                 (cx + cw * .36, cy - ch * .37), (cx - cw * .36, cy - ch * .37)], "ink", .18, edge=False, zorder=7)
        c.ax.plot([cx - cw * .35, cx + cw * .35], [cy - ch * .15, cy - ch * .15], color="white", lw=.65, zorder=8)
    transform = Affine2D()
    if crop is not None:
        u0, v0, u1, v1 = crop
        transform.translate(-x - u0 * w, -y - v0 * h).scale(1 / (u1 - u0), 1 / (v1 - v0)).translate(x, y)
    if mirror:
        transform.scale(-1, 1).translate(2 * x + w, 0)
    viewport = Rectangle((x, y), w, h, transform=c.ax.transData)
    for artist in set(c.ax.get_children()) - original_artists:
        # Patches retain their local dimensions. Only their data transform is
        # replaced; the clip rectangle stays in the final image viewport.
        artist.set_transform(transform + c.ax.transData)
        artist.set_clip_path(viewport)
    if label:
        c.text(x + w / 2, y + h + 15, label, 11, "muted", ha="center")


def surround_scene(c, x, y, radius=100):
    """Illustrate the six named camera directions around an ego vehicle."""
    c.ax.add_patch(Circle((x, y), radius * 1.05, facecolor=tint("gray", .94), edgecolor=color("line"), lw=.8, zorder=1))
    c.ax.add_patch(Rectangle((x - radius * .34, y - radius), radius * .68, radius * 2,
                             facecolor=tint("gray", .75), edgecolor="none", zorder=1))
    for offset in (-.64, .62):
        c.ax.plot([x, x], [y + radius * offset, y + radius * (offset + .18)], color="white", lw=2, zorder=2)
    labels = ("F", "FR", "BR", "B", "BL", "FL")
    for index, angle in enumerate((-90, -30, 30, 90, 150, 210)):
        a = np.deg2rad(angle)
        near = np.array([x, y]) + np.array([np.cos(a), np.sin(a)]) * 23
        far = [np.array([x, y]) + radius * np.array([np.cos(a + delta), np.sin(a + delta)]) for delta in (-.37, .37)]
        tone = "blue" if index % 2 == 0 else "purple"
        face(c, [near, *far], tone, .62, alpha=.46, zorder=2)
        c.ax.plot([near[0], far[0][0]], [near[1], far[0][1]], color=color(tone), lw=.75, zorder=3)
        c.ax.plot([near[0], far[1][0]], [near[1], far[1][1]], color=color(tone), lw=.75, zorder=3)
        c.ax.add_patch(Circle(near, 3, facecolor=color(tone), edgecolor="white", lw=.6, zorder=8))
        c.text(x + (radius + 18) * np.cos(a), y + (radius + 18) * np.sin(a), labels[index], 11, tone, True, "center")
    vehicle(c, x, y, .7)


def texture(rows=6, cols=9, phase=0):
    """Deterministic illustrative feature values, not learned activations."""
    r, col = np.mgrid[:rows, :cols]
    return np.clip(.42 + .23 * np.sin(col * .85 + phase) * np.cos(r * .95)
                   + .35 * np.exp(-((r - rows * .55)**2 + (col - cols * .6)**2) / 5), 0, 1)


def occupancy():
    values = np.zeros((10, 10))
    values[2:4, 2:5] = 1
    values[6:8, 5:8] = .85
    values[1:3, 7:9] = .65
    return values


def volume(c, x, y, w, h, depth=24, tone="blue", values=None, selected=None):
    """A textured tensor cuboid: front=image sites, extrusion=feature channels."""
    shift = np.array([depth, -depth * .55])
    a, b, d, e = (np.array(p) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)))
    face(c, [a, b, b + shift, a + shift], tone, .48, zorder=3)
    face(c, [b, d, d + shift, b + shift], tone, .22, zorder=3)
    face(c, [a, b, d, e], tone, .88, zorder=4)
    if values is not None:
        rows, cols = np.shape(values)
        for r in range(rows):
            for col in range(cols):
                c.ax.add_patch(Rectangle((x + col * w / cols, y + r * h / rows), w / cols, h / rows,
                                         facecolor=tint(tone, .94 - .77 * float(values[r][col])),
                                         edgecolor=tint(tone, .8), lw=.2, zorder=5))
        if selected is not None:
            r, col = selected
            c.ax.add_patch(Rectangle((x + col * w / cols, y + r * h / rows), w / cols, h / rows,
                                     facecolor="none", edgecolor=color("red"), lw=2.2, zorder=7))
    # Channel slices along the top/right face make extrusion explicit.
    for fraction in (.25, .5, .75):
        p, q = b + shift * fraction, d + shift * fraction
        c.ax.plot([p[0], q[0]], [p[1], q[1]], color=tint(tone, .7), lw=.65, zorder=6)


def ground_grid(c, x, y, w, h, values, tone="teal", pillars=0):
    """Ego x runs right; ego y runs back along the oblique plane."""
    values = np.asarray(values)
    nx, ny = values.shape

    def point(u, v, z=0):
        return np.array([x + u * w + v * w * .45, y - v * h - z])

    for v in reversed(range(ny)):
        for u in range(nx):
            value = float(values[u, v])
            corners = [point(u / nx, v / ny), point((u + 1) / nx, v / ny),
                       point((u + 1) / nx, (v + 1) / ny), point(u / nx, (v + 1) / ny)]
            face(c, corners, tone, .96 - .71 * value, zorder=2)
            if pillars and value > .2:
                height = value * pillars
                top = [p - [0, height] for p in corners]
                face(c, [corners[0], corners[1], top[1], top[0]], tone, .45, alpha=.9, zorder=3)
                face(c, [corners[1], corners[2], top[2], top[1]], tone, .2, alpha=.9, zorder=3)
                face(c, top, tone, .7, alpha=.95, zorder=4)
    return point


def frustum_volume(c, origin, axis, width=130, height=150, layers=4, tone="purple", selected=True):
    origin, axis = np.array(origin), np.array(axis)
    scales = np.linspace(.32, 1, layers)
    far = None
    for scale in reversed(scales):
        center = origin + axis * scale
        horizontal = np.array([width / 2, 0]) * scale
        vertical = np.array([height * .12, height / 2]) * scale
        corners = [center - horizontal - vertical, center + horizontal - vertical,
                   center + horizontal + vertical, center - horizontal + vertical]
        face(c, corners, tone, .79, alpha=.58, zorder=2)
        for fraction in (-.5, 0, .5):
            p = center + horizontal * fraction
            c.ax.plot([(p - vertical)[0], (p + vertical)[0]],
                      [(p - vertical)[1], (p + vertical)[1]], color=tint(tone, .36), lw=.45, zorder=3)
        if far is None:
            far = corners
    for corner in far:
        c.ax.plot([origin[0], corner[0]], [origin[1], corner[1]], color=color(tone), lw=.8, zorder=3)
    if selected:
        ray = axis + np.array([width * .14, height * .08])
        c.ax.plot([origin[0], (origin + ray)[0]], [origin[1], (origin + ray)[1]], color=color("red"), lw=1.8, zorder=6)
        for scale in scales:
            c.ax.add_patch(Circle(origin + ray * scale, 4, facecolor=color("red"), edgecolor="white", lw=.5, zorder=7))
    c.ax.add_patch(Circle(origin, 5, facecolor=color("blue"), edgecolor="white", lw=.8, zorder=8))


def callout(c, x, y, number, text, tone="teal", size=14):
    c.ax.add_patch(Circle((x, y), 12, facecolor=color(tone), edgecolor="none", zorder=8))
    c.text(x, y, number, 11, "white", True, "center", zorder=9)
    c.text(x + 21, y, text, size, tone, bold=True)

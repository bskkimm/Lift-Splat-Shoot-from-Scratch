"""Vector drawing vocabulary for the original LSS illustrations.

The design uses spatial feature stacks, worked tensor entries and geometric
examples. All coordinates are in a 1400 x 800 layout, independent of raster DPI.
"""

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
from PIL import Image


COLORS = {
    "ink": "#162C43", "muted": "#52677C", "line": "#CBD6E2", "paper": "#F5F8FC",
    "blue": "#2266AE", "teal": "#087F83", "purple": "#7750A9", "gold": "#A56A08",
    "red": "#C44B5A", "gray": "#6E7F91", "white": "#FFFFFF",
    "pale_blue": "#E9F2FC", "pale_teal": "#E6F5F3", "pale_purple": "#F0EAF9",
    "pale_gold": "#FFF4D9", "pale_red": "#FCECEF", "pale_gray": "#EDF1F5",
}


def color(name):
    return COLORS.get(name, name)


class Canvas:
    def __init__(self, number, title, subtitle, category="PRACTICAL LSS"):
        with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                             "svg.hashsalt": "lss-architecture-v1", "font.size": 16}):
            self.figure = plt.figure(figsize=(14, 8), dpi=100)
        self.ax = self.figure.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, 1400), ylim=(800, 0))
        self.ax.axis("off")
        self.card_labels = []
        self.text(34, 42, f"{number:02d}", 26, "teal", bold=True)
        self.text(94, 42, title, 25, bold=True)
        self.text(34, 86, subtitle, 15, "muted")
        self.ax.plot([34, 1366], [112, 112], color=color("line"), lw=.8)
        self.text(1366, 765, category, 11, "muted", ha="right")

    def text(self, x, y, text, size=16, tone="ink", bold=False, ha="left", fit_width=None, **kwargs):
        if fit_width is not None:
            width = self.width(str(text), size, bold)
            if width > fit_width:
                size *= fit_width / width
                if size < 12:
                    raise ValueError(f"shorten label rather than shrinking below 12 pt: {text}")
        return self.ax.text(x, y, str(text), fontsize=size, color=color(tone),
                            weight="bold" if bold else "normal", ha=ha, va="center",
                            linespacing=1.45, fontfamily="DejaVu Sans", **kwargs)

    def width(self, text, size, bold=False):
        properties = FontProperties(family="DejaVu Sans", size=size, weight="bold" if bold else "normal")
        renderer = self.figure.canvas.get_renderer()
        return max(renderer.get_text_width_height_descent(line, properties, False)[0]
                   for line in text.split("\n"))

    def wrap(self, text, width, size=15):
        lines = []
        for paragraph in text.split("\n"):
            line = ""
            for word in paragraph.split():
                candidate = (line + " " + word).strip()
                if line and self.width(candidate, size) > width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        return "\n".join(lines)

    def panel(self, x, y, w, h, title="", subtitle="", tone="blue"):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                         facecolor="white", edgecolor=color("line"), lw=1))
        if title:
            self.text(x + 20, y + 30, title, 18, tone, bold=True, fit_width=w - 40)
        if subtitle:
            self.text(x + 20, y + 60, subtitle, 14, "muted", fit_width=w - 40)

    def card(self, x, y, w, h, title, body="", tone="blue", size=18):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                                         facecolor=color("pale_" + tone), edgecolor=color(tone), lw=1.2))
        title_label = self.text(x + w / 2, y + (h / 2 if not body else 26), title, size, tone, True, "center", fit_width=w - 24)
        labels = [title_label]
        if body:
            renderer = self.figure.canvas.get_renderer()
            title_height = title_label.get_window_extent(renderer).height
            title_y = y + max(26, title_height / 2 + 8)
            title_label.set_position((x + w / 2, title_y))
            body_top, body_bottom = title_y + title_height / 2 + 7, y + h - 7
            body_label = self.text(x + w / 2, (body_top + body_bottom) / 2,
                                   self.wrap(body, w - 28), 15, ha="center")
            body_height = body_label.get_window_extent(renderer).height
            if body_height > body_bottom - body_top:
                body_size = 15 * (body_bottom - body_top) / body_height * .98
                if body_size < 12:
                    raise ValueError(f"shorten card body: {title}")
                body_label.set_fontsize(body_size)
            labels.append(body_label)
        self.card_labels.append((labels, (x, y, x + w, y + h)))

    def arrow(self, start, end, label="", tone="teal", curve=0, dashed=False):
        self.ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                         linewidth=1.8, color=color(tone),
                                         connectionstyle=f"arc3,rad={curve}",
                                         linestyle="--" if dashed else "-", shrinkA=4, shrinkB=4))
        if label:
            self.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - 17,
                      label, 13, tone, ha="center")

    def path(self, points, tone="teal", dashed=False, width=2):
        self.ax.plot(*zip(*points), color=color(tone), lw=width, ls="--" if dashed else "-",
                     solid_capstyle="round")

    def matrix(self, x, y, values, cell=(66, 48), tone="teal", rows=None, cols=None,
               highlight=(), disabled=()):
        w, h = cell
        for r, row in enumerate(values):
            if rows:
                self.text(x - 12, y + (r + .5) * h, rows[r], 14, "muted", ha="right")
            for c, value in enumerate(row):
                inactive = (r, c) in disabled
                focus = (r, c) in highlight
                self.ax.add_patch(Rectangle((x + c * w, y + r * h), w, h,
                                            facecolor=color("pale_gray" if inactive else "pale_" + tone),
                                            edgecolor=color(tone) if focus else "white", lw=2.4 if focus else 1.5))
                value = f"{value:g}" if isinstance(value, (int, float, np.number)) else value
                self.text(x + (c + .5) * w, y + (r + .5) * h, value, 16,
                          "muted" if inactive else "ink", bold=focus, ha="center")
        if cols:
            for c, label in enumerate(cols):
                self.text(x + (c + .5) * w, y - 20, label, 14, "muted", ha="center")

    def stack(self, x, y, w, h, tone="purple", layers=4, label=""):
        for i in reversed(range(layers)):
            dx, dy = i * 10, -i * 8
            self.ax.add_patch(Rectangle((x + dx, y + dy), w, h, facecolor=color("pale_" + tone),
                                        edgecolor=color(tone), lw=1, alpha=.95))
        for step in range(1, 4):
            self.path([(x + step * w / 4, y), (x + step * w / 4, y + h)], tone, width=.6)
            self.path([(x, y + step * h / 4), (x + w, y + step * h / 4)], tone, width=.6)
        if label:
            self.text(x + w / 2, y + h + 25, label, 14, tone, ha="center")

    def camera(self, x, y, w, h, label="", tone="blue"):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=color("pale_blue"), edgecolor=color(tone), lw=1.2))
        self.ax.add_patch(Polygon([(x, y + h), (x + .4 * w, y + .3 * h),
                                  (x + .6 * w, y + .3 * h), (x + w, y + h)],
                                 facecolor=color("pale_gray"), edgecolor="none"))
        self.path([(x + .5 * w, y + .9 * h), (x + .5 * w, y + .75 * h)], "gold", width=2)
        self.path([(x + .5 * w, y + .62 * h), (x + .5 * w, y + .52 * h)], "gold", width=1.3)
        self.ax.add_patch(FancyBboxPatch((x + .65 * w, y + .52 * h), .18 * w, .22 * h,
                                         boxstyle="round,pad=0,rounding_size=3",
                                         facecolor=color(tone), edgecolor="white", lw=.7))
        if label:
            self.text(x + w / 2, y + h + 17, label, 12, tone, ha="center")

    def plane(self, x, y, w, h, tone="teal", cells=5):
        """Oblique BEV grid, useful for showing the vertical-to-BEV reduction."""
        a = np.array([x, y]); b = np.array([w, 0]); d = np.array([.35 * w, -h])
        self.ax.add_patch(Polygon([a, a + b, a + b + d, a + d], facecolor=color("pale_" + tone),
                                 edgecolor=color(tone), lw=1.4))
        for i in range(1, cells):
            self.path([a + b * i / cells, a + b * i / cells + d], tone, width=.7)
            self.path([a + d * i / cells, a + d * i / cells + b], tone, width=.7)

    def note(self, text):
        self.ax.plot([34, 1366], [732, 732], color=color("line"), lw=.8)
        self.text(34, 765, text, 12, "muted")

    def save(self, directory, slug):
        self.figure.canvas.draw()
        renderer = self.figure.canvas.get_renderer()
        for text in self.ax.texts:
            box = text.get_window_extent(renderer)
            if box.x0 < 8 or box.y0 < 8 or box.x1 > 1392 or box.y1 > 792:
                raise ValueError(f"{slug}: text outside canvas: {text.get_text()!r}")
        for labels, (left, top, right, bottom) in self.card_labels:
            boxes = [label.get_window_extent(renderer) for label in labels]
            for label, box in zip(labels, boxes):
                if box.x0 < left + 4 or box.x1 > right - 4 or box.y0 < 800 - bottom + 4 or box.y1 > 800 - top - 4:
                    raise ValueError(f"{slug}: text outside its card: {label.get_text()!r}")
            if len(boxes) == 2 and boxes[0].overlaps(boxes[1]):
                raise ValueError(f"{slug}: card title overlaps body: {labels[0].get_text()!r}")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "lss-architecture-v1"}):
            self.figure.savefig(directory / f"{slug}.svg", facecolor="white", metadata={"Date": None})
        svg = directory / f"{slug}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        buffer = BytesIO()
        self.figure.savefig(buffer, format="png", dpi=75, facecolor="white", metadata={"Software": "LSS architecture drawings"})
        with Image.open(buffer) as image:
            image.convert("RGB").quantize(colors=24, dither=Image.Dither.NONE).save(directory / f"{slug}.png", optimize=True)
        plt.close(self.figure)

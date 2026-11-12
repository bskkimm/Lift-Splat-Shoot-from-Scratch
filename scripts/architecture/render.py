"""Render original assets and inventory their provenance without importing LSS."""

import ast
import hashlib
import importlib
import json
from pathlib import Path

import matplotlib
from matplotlib import font_manager, ft2font
import numpy as np
import PIL

from .examples import worked_examples
from .inventory import FIGURES


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_digest(notebook):
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    return hashlib.sha256(ast.dump(ast.parse(code), include_attributes=False).encode()).hexdigest()


def generate(directory, notebook_path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    examples = worked_examples()
    figures = []
    for entry in FIGURES:
        module_name, function_name = entry["draw"].split(".")
        module = importlib.import_module(f"scripts.architecture.{module_name}")
        figure = getattr(module, function_name)(examples)
        figure.save(directory, entry["id"])
        figures.append({
            "id": entry["id"], "anchor": "fig-" + entry["id"],
            "position": "after" if "after" in entry else "before",
            "lesson_marker": entry.get("after", entry.get("before")),
            "neighbor_type": "markdown" if entry.get("before", "").startswith("#") else "code",
            "title": entry["title"], "alt": entry["title"] + ". " + entry["caption"],
            "caption": entry["caption"],
            "png": entry["id"] + ".png", "svg": entry["id"] + ".svg",
            "png_sha256": sha256(directory / (entry["id"] + ".png")),
            "svg_sha256": sha256(directory / (entry["id"] + ".svg")),
            "sources": {source: sha256(ROOT / source) for source in entry["sources"]},
        })
    font = Path(font_manager.findfont("DejaVu Sans"))
    generators = sorted(Path(__file__).parent.glob("*.py")) + [ROOT / "scripts/illustrate_notebook.py"]
    manifest = {
        "format_version": 1,
        "reference_inspiration": "PlanTF_from_Scratch@2e02a5c485ef2edd88555610143828a7a122c96f; original LSS artwork",
        "visualization_standard": "paper-reimplementation-playbook@707c64b138d5e6c534af3d197fd4f278aab03d45",
        "settings": "practical full-size segmentation shapes; numerical examples explicitly reduced/illustrative",
        "visual_design": "v2: original street/vehicle illustrations, spatial feature volumes and geometric evidence routing",
        "png_palette": "eight fixed semantic colors plus sixteen adaptive colors; no dithering",
        "environment": {"matplotlib": matplotlib.__version__, "pillow": PIL.__version__,
                        "numpy": np.__version__, "freetype": ft2font.__freetype_version__,
                        "font": font.name, "font_sha256": sha256(font), "png_size": [1050, 600]},
        "generators": {str(path.relative_to(ROOT)): sha256(path) for path in generators},
        "notebook_code_ast_sha256": code_digest(json.loads(Path(notebook_path).read_text())),
        "worked_examples": examples,
        "figures": figures,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

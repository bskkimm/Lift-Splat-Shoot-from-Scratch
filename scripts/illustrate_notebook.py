"""Render, embed and validate original LSS notebook architecture illustrations.

Commands run from any working directory. Rendering is a development operation;
the notebook itself displays local Markdown attachments without this script.
"""

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.architecture.inventory import FIGURES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_ast(notebook):
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    return ast.dump(ast.parse(code), include_attributes=False)


def markdown_alt(text):
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def validate_assets(assets, manifest):
    ids = [entry["id"] for entry in manifest["figures"]]
    expected = {entry["id"] for entry in FIGURES}
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError("figure inventory must contain exactly the fourteen distinct approved figures")
    for entry in manifest["figures"]:
        spec = next(item for item in FIGURES if item["id"] == entry["id"])
        if (entry["title"] != spec["title"] or entry["caption"] != spec["caption"]
                or entry["anchor"] != "fig-" + entry["id"]
                or entry["lesson_marker"] != spec.get("before", spec.get("after"))
                or entry["neighbor_type"] != ("markdown" if spec.get("before", "").startswith("#") else "code")
                or entry["position"] != ("before" if "before" in spec else "after")):
            raise ValueError(f"{entry['id']}: stale caption, anchor or placement")
        if set(entry["sources"]) != set(spec["sources"]):
            raise ValueError(f"{entry['id']}: incomplete source coverage")
        for suffix in ("png", "svg"):
            if entry[suffix] != entry["id"] + "." + suffix:
                raise ValueError(f"{entry['id']}: unexpected asset filename")
            path = assets / entry[suffix]
            if not path.is_file() or digest(path) != entry[suffix + "_sha256"]:
                raise ValueError(f"{entry['id']}: missing asset or {suffix} hash mismatch")
        with Image.open(assets / entry["png"]) as image:
            image.verify()
        for source, checksum in entry["sources"].items():
            if digest(ROOT / source) != checksum:
                raise ValueError(f"{entry['id']}: source changed: {source}; regenerate and inspect")
    generators = {str(path.relative_to(ROOT)) for path in (ROOT / "scripts/architecture").glob("*.py")}
    generators.add("scripts/illustrate_notebook.py")
    if set(manifest["generators"]) != generators:
        raise ValueError("incomplete drawing source coverage")
    for source, checksum in manifest["generators"].items():
        if digest(ROOT / source) != checksum:
            raise ValueError(f"drawing source changed: {source}; regenerate and inspect")
    expected_assets = {entry[k] for entry in manifest["figures"] for k in ("png", "svg")}
    actual_assets = {path.name for path in assets.iterdir() if path.suffix in (".png", ".svg")}
    if expected_assets != actual_assets:
        raise ValueError("orphaned or missing PNG/SVG files")


def attach(notebook_path, assets, manifest):
    validate_assets(assets, manifest)
    notebook = json.loads(notebook_path.read_text())
    original_ast = code_ast(notebook)
    specs = manifest["figures"]
    before_markers = [entry["lesson_marker"] for entry in specs if entry["position"] == "before"]
    # Split only at a top-level Python statement boundary, never inside a class.
    expanded = []
    for cell in notebook["cells"]:
        if cell.get("metadata", {}).get("architecture_figure"):
            continue
        if cell["cell_type"] != "code":
            expanded.append(cell)
            continue
        source = "".join(cell["source"])
        lines = source.splitlines(keepends=True)
        boundaries = {0, len(lines)}
        for node in ast.parse(source).body:
            if any(lines[node.lineno - 1].startswith(marker) for marker in before_markers):
                boundaries.add(node.lineno - 1)
        boundaries = sorted(boundaries)
        for first, last in zip(boundaries, boundaries[1:]):
            part = "".join(lines[first:last])
            if not part.strip():
                continue
            expanded.append({**cell, "source": part.splitlines(keepends=True),
                             "id": "code-" + hashlib.sha256(part.encode()).hexdigest()[:16]})
    cells, found = [], []

    def figure_cell(entry):
        raw = (assets / entry["png"]).read_bytes()
        source = f"### {entry['title']}\n\n![{markdown_alt(entry['alt'])}](attachment:{entry['png']})\n\n*{entry['caption']}*\n"
        return {"cell_type": "markdown", "id": entry["anchor"],
                "metadata": {"architecture_figure": entry["id"]},
                "attachments": {entry["png"]: {"image/png": base64.b64encode(raw).decode()}},
                "source": source.splitlines(keepends=True)}

    for cell in expanded:
        source = "".join(cell["source"])
        matches = [entry for entry in specs if cell["cell_type"] == entry["neighbor_type"]
                   and source.lstrip().startswith(entry["lesson_marker"])]
        for entry in matches:
            if entry["position"] == "before":
                cells.append(figure_cell(entry)); found.append(entry["id"])
        if "id" not in cell:
            cell["id"] = "text-" + hashlib.sha256(source.encode()).hexdigest()[:16]
        cells.append(cell)
        for entry in matches:
            if entry["position"] == "after":
                cells.append(figure_cell(entry)); found.append(entry["id"])
    if len(found) != len(specs) or set(found) != {entry["id"] for entry in specs}:
        raise ValueError(f"could not uniquely place every figure; found {found}")
    notebook["cells"] = cells
    if code_ast(notebook) != original_ast:
        raise ValueError("attachment insertion changed notebook Python code")
    encoded = json.dumps(notebook, ensure_ascii=False, separators=(",", ":"), indent=1) + "\n"
    if len(encoded.encode()) >= 500 * 1024:
        raise ValueError("illustrated notebook exceeds the existing 500 KiB file limit; compress previews")
    notebook_path.write_text(encoded)


def check(notebook_path, assets, manifest):
    validate_assets(assets, manifest)
    notebook = json.loads(notebook_path.read_text())
    if hashlib.sha256(code_ast(notebook).encode()).hexdigest() != manifest["notebook_code_ast_sha256"]:
        raise ValueError("notebook code changed; regenerate figures and revalidate parity")
    if notebook_path.stat().st_size >= 500 * 1024:
        raise ValueError("notebook exceeds the existing 500 KiB limit")
    all_ids = [cell["id"] for cell in notebook["cells"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate notebook cell IDs")
    for entry in manifest["figures"]:
        if all_ids.count(entry["anchor"]) != 1:
            raise ValueError(f"{entry['id']}: missing or repeated notebook anchor")
        index = all_ids.index(entry["anchor"])
        cell = notebook["cells"][index]
        source = "".join(cell["source"])
        references = re.findall(r"!\[(?:\\.|[^\]\\])*\]\(attachment:([^)]+)\)", source)
        if references != [entry["png"]] or set(cell.get("attachments", {})) != {entry["png"]}:
            raise ValueError(f"{entry['id']}: missing, duplicate or orphaned attachment")
        attachment = base64.b64decode(cell["attachments"][entry["png"]]["image/png"], validate=True)
        if hashlib.sha256(attachment).hexdigest() != entry["png_sha256"]:
            raise ValueError(f"{entry['id']}: notebook attachment hash mismatch")
        with Image.open(io.BytesIO(attachment)) as image:
            image.verify()
        if markdown_alt(entry["alt"]) not in source or entry["caption"] not in source:
            raise ValueError(f"{entry['id']}: missing caption or alt text")
        neighbor = index + (1 if entry["position"] == "before" else -1)
        lesson = notebook["cells"][neighbor]
        if lesson["cell_type"] != entry["neighbor_type"] or not "".join(lesson["source"]).lstrip().startswith(entry["lesson_marker"]):
            raise ValueError(f"{entry['id']}: figure is not adjacent to its inventoried lesson")
    known = {entry["anchor"] for entry in manifest["figures"]}
    for cell in notebook["cells"]:
        if cell.get("attachments") and cell["id"] not in known:
            raise ValueError("uninventoried notebook image attachment")


def contact_sheet(assets, manifest, output):
    canvas = Image.new("RGB", (1120, 320 * 7), "white")
    for index, entry in enumerate(manifest["figures"]):
        with Image.open(assets / entry["png"]) as image:
            preview = image.convert("RGB").resize((560, 320), Image.Resampling.LANCZOS)
        canvas.paste(preview, ((index % 2) * 560, (index // 2) * 320))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render", action="store_true")
    mode.add_argument("--attach", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--assets", type=Path, default=ROOT / "notebooks/assets/architecture")
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/lss_e2e_walkthrough.ipynb")
    args = parser.parse_args()
    try:
        if args.render:
            from scripts.architecture.render import generate
            generate(args.assets, args.notebook)
            print("Rendered 14 original architecture figures (PNG + editable SVG).")
            return
        manifest = json.loads((args.assets / "manifest.json").read_text())
        if args.attach:
            attach(args.notebook, args.assets, manifest)
        elif args.check:
            check(args.notebook, args.assets, manifest)
        else:
            contact_sheet(args.assets, manifest, args.contact_sheet)
        print("Architecture illustration contract passed: 14 figures.")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Illustration error: {exc}\n")


if __name__ == "__main__":
    main()

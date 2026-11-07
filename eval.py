"""Evaluate a practical LSS segmentation checkpoint on a named nuScenes split."""

import argparse
import json
import os
import pickle
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from lss.data.loader import build_loader
from lss.data.nuscenes_dataset import NuScenesCameraDataset
from lss.data.recipes import dataset_options
from lss.models.segmentation import LSSSegmentation
from lss.reproducibility import configure_cuda_workspace, seed_everything
from lss.segmentation_engine import run_segmentation_epoch


def write_json_once(path, payload):
    """Atomically create a result file, never replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a practical LSS BEV segmentation checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataroot", default=None)
    parser.add_argument("--version", choices=("v1.0-mini", "v1.0-trainval"), default=None)
    parser.add_argument("--depth-bins", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cpu_threads <= 0 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error("cpu-threads and batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("max-samples must be positive")
    if args.output_dir is not None and args.output is None:
        parser.error("--output-dir requires --output")
    output = None
    if args.output is not None:
        output = Path(args.output).expanduser()
        if args.output_dir is not None:
            output = Path(args.output_dir).expanduser() / output
        if output.suffix.lower() != ".json":
            parser.error("--output must name a JSON file")
        if output.exists():
            parser.error(f"output already exists: {output}")
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint file not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (not isinstance(state, dict) or state.get("format_version") != 1
                or state.get("types", {}).get("model") != "lss.models.segmentation.LSSSegmentation"):
            raise ValueError("unsupported model or checkpoint format")
        config = json.loads(state["config_json"])
        if not isinstance(config, dict) or config.get("recipe") not in ("practical-resize-fp32-v1", "reference-data-fp32-v1"):
            raise ValueError("unsupported training recipe")
        required = ("version", "seed", "x_bounds", "y_bounds", "z_bounds", "depth_bounds",
                    "context_channels", "out_channels", "dataroot", "image_size", "batch_size", "pos_weight")
        missing = [name for name in required if name not in config]
        if missing or "model" not in state or "epoch" not in state:
            raise ValueError(f"missing checkpoint fields: {missing}")
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, EOFError, pickle.UnpicklingError) as exc:
        parser.error(f"incompatible checkpoint: {exc}")
    if args.version is not None and args.version != config["version"]:
        parser.error("--version must match the checkpoint configuration")
    seed = config["seed"] if args.seed is None else args.seed
    if seed < 0 or seed >= 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    if args.device == "cuda":
        try:
            configure_cuda_workspace()
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    seed_everything(seed)
    np.random.seed(seed)
    model = LSSSegmentation(x_bounds=config["x_bounds"], y_bounds=config["y_bounds"],
                            z_bounds=config["z_bounds"], depth_bounds=config["depth_bounds"],
                            context_channels=config["context_channels"], out_channels=config["out_channels"],
                            pretrained=False)
    if args.depth_bins is not None and args.depth_bins != model.depth_values.numel():
        parser.error("--depth-bins must match the checkpoint configuration")
    try:
        model.load_state_dict(state["model"], strict=True)
    except (KeyError, RuntimeError, ValueError) as exc:
        parser.error(f"incompatible checkpoint model: {exc}")
    device = torch.device(args.device)
    model.to(device).eval()
    dataroot = Path(args.dataroot or config["dataroot"]).expanduser().resolve()
    dataset = NuScenesCameraDataset(dataroot=dataroot, version=config["version"], split="val",
                                   bev_bounds=(config["x_bounds"], config["y_bounds"]),
                                   **dataset_options(config, training=False))
    total_samples = len(dataset)
    if total_samples == 0:
        parser.error("validation split contains no samples")
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, total_samples)))
    loader = build_loader(dataset, batch_size=args.batch_size or config["batch_size"],
                          shuffle=False, num_workers=0)
    result = run_segmentation_epoch(model, loader, device, pos_weight=config["pos_weight"])
    payload = {
        "metric_name": "practical_bev_vehicle_iou",
        "scope": "subset" if args.max_samples is not None else "full_split",
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": state["epoch"],
        "recipe": config["recipe"], "dataroot": str(dataroot), "version": config["version"],
        "split": "val", "total_split_samples": total_samples, "max_samples": args.max_samples,
        "device": args.device, "cpu_threads": args.cpu_threads, "seed": seed, "metrics": asdict(result),
    }
    if output is not None:
        write_json_once(output, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

"""Practical, single-process nuScenes BEV segmentation training entry point."""

import argparse
import json
import math
import os
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from lss.data.loader import build_loader
from lss.data.nuscenes_dataset import NuScenesCameraDataset
from lss.models.segmentation import LSSSegmentation
from lss.reproducibility import configure_cuda_workspace, seed_everything
from lss.segmentation_engine import run_segmentation_epoch
from lss.training_checkpoint import load_epoch_checkpoint, save_epoch_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description="Train the practical LSS BEV segmentation recipe")
    parser.add_argument("--dataroot", default=None)
    parser.add_argument("--version", choices=("v1.0-mini", "v1.0-trainval"), default="v1.0-trainval")
    parser.add_argument("--epochs", type=int, default=1, help="total target epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--depth-bins", type=int, default=None, help="override bin count starting at --depth-bound minimum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--checkpoint-name", default="last_checkpoint.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(128, 352))
    parser.add_argument("--x-bound", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(-50, 50, 0.5))
    parser.add_argument("--y-bound", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(-50, 50, 0.5))
    parser.add_argument("--z-bound", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(-10, 10, 20))
    parser.add_argument("--depth-bound", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"), default=(4, 45, 1))
    parser.add_argument("--context-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--scheduler", choices=("none",), default="none")
    return parser


def resolved_config(args, depth_bounds):
    return {
        "recipe": "practical-resize-fp32-v1",
        "dataroot": str(Path(args.dataroot).expanduser().resolve()),
        "version": args.version,
        "train_split": "train", "validation_split": "val",
        "image_size": list(args.image_size),
        "x_bounds": list(args.x_bound), "y_bounds": list(args.y_bound),
        "z_bounds": list(args.z_bound), "depth_bounds": list(depth_bounds),
        "context_channels": args.context_channels, "out_channels": 1,
        "pretrained": False, "batch_size": args.batch_size,
        "workers": 0, "shuffle": True, "seed": args.seed, "device": args.device,
        "cpu_threads": args.cpu_threads, "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if args.device == "cuda" else None,
        "optimizer": "Adam", "lr": args.lr, "weight_decay": args.weight_decay,
        "scheduler": args.scheduler, "pos_weight": args.pos_weight,
        "grad_clip_norm": args.grad_clip_norm, "precision": "float32",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torchvision": torchvision.__version__, "numpy": np.__version__,
                        "opencv": cv2.__version__, "nuscenes_devkit": version("nuscenes-devkit"),
                        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.context_channels <= 0 or args.cpu_threads <= 0:
        parser.error("epochs, batch-size, context-channels and cpu-threads must be positive")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    if not all(math.isfinite(value) for value in (args.lr, args.weight_decay, args.pos_weight, args.grad_clip_norm)):
        parser.error("optimizer and loss settings must be finite")
    if args.lr <= 0 or args.weight_decay < 0 or args.pos_weight <= 0 or args.grad_clip_norm <= 0:
        parser.error("learning rate, positive weight and gradient clip must be positive; weight decay must be nonnegative")
    if not args.dry_run and args.dataroot is None:
        parser.error("--dataroot is required for training")
    checkpoint_path = Path(args.output_dir).expanduser() / args.checkpoint_name
    if not args.dry_run and checkpoint_path.exists():
        if args.resume is None or Path(args.resume).expanduser().resolve() != checkpoint_path.resolve():
            parser.error(f"checkpoint already exists: {checkpoint_path}; resume it or choose a different output")
    if args.device == "cuda":
        try:
            configure_cuda_workspace()
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        if not torch.cuda.is_available():
            parser.error("CUDA was requested but is unavailable")
    depth_bounds = tuple(args.depth_bound)
    if args.depth_bins is not None:
        if args.depth_bins <= 0:
            parser.error("depth-bins must be positive")
        depth_bounds = (depth_bounds[0], depth_bounds[0] + args.depth_bins * depth_bounds[2], depth_bounds[2])
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    seed_everything(args.seed)
    np.random.seed(args.seed)
    model = LSSSegmentation(x_bounds=args.x_bound, y_bounds=args.y_bound, z_bounds=args.z_bound,
                            depth_bounds=depth_bounds, context_channels=args.context_channels, pretrained=False)
    if args.dry_run:
        print(f"model construction only; parameters={sum(p.numel() for p in model.parameters())}; dry_run=True")
        return

    device = torch.device(args.device)
    model.to(device)
    config = resolved_config(args, depth_bounds)
    dataset_args = {"dataroot": args.dataroot, "version": args.version, "image_size": tuple(args.image_size),
                    "bev_bounds": (args.x_bound, args.y_bound)}
    train_data = NuScenesCameraDataset(**dataset_args, split="train")
    validation_data = NuScenesCameraDataset(**dataset_args, split="val")
    if len(train_data) == 0:
        parser.error("training split contains no samples")
    if len(validation_data) == 0:
        parser.error("validation split contains no samples")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = build_loader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    validation_loader = build_loader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    completed_epoch = 0
    if args.resume is not None:
        completed_epoch = load_epoch_checkpoint(args.resume, model, optimizer, config, generator)
    if completed_epoch >= args.epochs:
        parser.error("--epochs must exceed the completed checkpoint epoch")
    for epoch in range(completed_epoch + 1, args.epochs + 1):
        train_result = run_segmentation_epoch(model, train_loader, device, optimizer=optimizer,
                                              pos_weight=args.pos_weight, grad_clip_norm=args.grad_clip_norm)
        validation_result = run_segmentation_epoch(model, validation_loader, device, pos_weight=args.pos_weight)
        if not math.isfinite(train_result.loss) or not math.isfinite(validation_result.loss):
            raise RuntimeError("non-finite segmentation loss; checkpoint not saved")
        save_epoch_checkpoint(checkpoint_path, model, optimizer, epoch, config, generator,
                              capture_cuda=device.type == "cuda")
        print(json.dumps({"recipe": config["recipe"], "epoch": epoch,
                          "train": asdict(train_result), "validation": asdict(validation_result)},
                         allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

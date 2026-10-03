import argparse
from pathlib import Path
import torch
from lss.models.lss import LSS
from lss.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train pure PyTorch LSS")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--dataroot", default="~/dataset/nuscenes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--depth-bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.seed is not None: seed_everything(args.seed)
    model = LSS(depth_bins=args.depth_bins)
    print(f"training LSS for {args.epochs} epochs from {args.dataroot}; parameters={sum(p.numel() for p in model.parameters())}")
    if not args.dry_run:
        output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), output_dir / "lss_model.pt")


if __name__ == "__main__": main()

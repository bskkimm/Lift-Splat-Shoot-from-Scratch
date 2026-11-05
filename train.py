import argparse
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
    parser.add_argument("--checkpoint-name", default="lss_model.pt")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("Training is not implemented yet; use --dry-run for model construction only.")
    if args.seed is not None: seed_everything(args.seed)
    model = LSS(depth_bins=args.depth_bins)
    print(f"model construction only; parameters={sum(p.numel() for p in model.parameters())}; dry_run={args.dry_run}")


if __name__ == "__main__": main()

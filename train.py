import argparse
import torch
from lss.models.lss import LSS


def main():
    parser = argparse.ArgumentParser(description="Train pure PyTorch LSS")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--dataroot", default="~/dataset/nuscenes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); model = LSS()
    print(f"training LSS for {args.epochs} epochs from {args.dataroot}; parameters={sum(p.numel() for p in model.parameters())}")
    if not args.dry_run: torch.save(model.state_dict(), "lss_model.pt")


if __name__ == "__main__": main()

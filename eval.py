import argparse
from lss.evaluation import export_predictions
from lss.models.lss import LSS
from lss.checkpoint import load_checkpoint
from lss.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Evaluate pure PyTorch LSS")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataroot", default="~/dataset/nuscenes")
    parser.add_argument("--depth-bins", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None: seed_everything(args.seed)
    model = LSS(depth_bins=args.depth_bins); load_checkpoint(model, args.checkpoint); model.eval(); print(f"evaluating {args.checkpoint} on {args.dataroot}")
    if args.output:
        if not args.output.endswith(".json"): raise ValueError("--output must be a .json file")
        export_predictions({"results": [], "meta": {}}, args.output)


if __name__ == "__main__": main()

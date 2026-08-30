import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate pure PyTorch LSS")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataroot", default="~/dataset/nuscenes")
    args = parser.parse_args(); print(f"evaluating {args.checkpoint} on {args.dataroot}")


if __name__ == "__main__": main()

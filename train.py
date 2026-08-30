import argparse


def main():
    parser = argparse.ArgumentParser(description="Train pure PyTorch LSS")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--dataroot", default="~/dataset/nuscenes")
    args = parser.parse_args(); print(f"training LSS for {args.epochs} epochs from {args.dataroot}")


if __name__ == "__main__": main()

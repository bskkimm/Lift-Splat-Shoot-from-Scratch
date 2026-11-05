import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate pure PyTorch LSS")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataroot", default="~/dataset/nuscenes")
    parser.add_argument("--depth-bins", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.parse_args()
    parser.error("Evaluation is not implemented yet; no predictions were produced.")


if __name__ == "__main__": main()

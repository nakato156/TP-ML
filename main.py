from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(description="A script to detect fraud.")
    parser.add_argument("input_file", type=Path, help="Path to the input file to be processed.")
    parser.add_argument("train", type=bool, help="Whether to train the model or not.")
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = args.input_file
    train = args.train



if __name__ == "__main__":
    main()
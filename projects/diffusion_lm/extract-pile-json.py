# to download
# wget https://mystic.the-eye.eu/public/AI/pile/train/00.jsonl.zst
# to unzip -> jsonl
# unzstd 00.jsonl.zst


from pathlib import Path

import os
from tqdm import tqdm
import argparse
import json

import random
import jsonlines
import numpy as np


def extract_json(in_path: str, out_path: str, train_ratio: float):
    os.makedirs(out_path, exist_ok=True)
    out_path = Path(out_path)

    with open(in_path, "r") as in_file, open(
        out_path.joinpath("train/00/a.jsonl"), "w"
    ) as out_train_file, open(
        out_path.joinpath("valid/00/a.jsonl"), "w"
    ) as out_valid_file:
        for line in tqdm(in_file):
            prob_train = random.uniform(0, 1)
            if prob_train < train_ratio:
                out_train_file.write(line)
            else:
                out_valid_file.write(line)


parser = argparse.ArgumentParser(description="Extract a .jsonl file")
parser.add_argument("in_path", type=str, help="input file (.jsonl) path")
parser.add_argument("--out_path", type=str, help="output file path")
parser.add_argument(
    "--train_ratio", type=float, default=0.9, help="portion of the set to keep"
)

args = parser.parse_args()

extract_json(args.in_path, args.out_path, args.train_ratio)
print("done")

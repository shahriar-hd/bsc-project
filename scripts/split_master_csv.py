"""
split_master_csv.py
Usage: python split_master_csv.py --input master.csv --output_dir splits/
"""

import argparse
import pandas as pd
from pathlib import Path


def split_master_csv(input_path: str, output_dir: str):
    df = pd.read_csv(input_path)

    required_cols = {"frame_path", "dataset", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    groups = df.groupby(["dataset", "split"])
    counts = {}

    for (dataset, split), group in groups:
        filename = f"{dataset}_{split}.csv"
        out_file = output_path / filename
        group.to_csv(out_file, index=False)
        counts[(dataset, split)] = len(group)
        print(f"  saved → {out_file}  ({len(group):,} rows)")

    print(f"\nTotal files written: {len(counts)}")
    print(f"Total rows processed: {len(df):,}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default="master.csv")
    p.add_argument("--output_dir", default="splits")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_master_csv(args.input, args.output_dir)

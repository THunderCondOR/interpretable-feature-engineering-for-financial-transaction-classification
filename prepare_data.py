"""Prepare benchmark datasets.

Input files:
    data/test_ids/gender_test_ids.csv
    data/test_ids/age_test_ids.csv
    data/test_ids/rosbank_test_ids.csv

Examples:
    python prepare_data.py --dataset all
    python prepare_data.py --dataset gender
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.prepare_data import VAL_SIZE_FROM_REMAINING, build_specs, prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare benchmark datasets.")
    parser.add_argument("--dataset", choices=["all", "gender", "age", "rosbank"], default="all")
    parser.add_argument("--test-ids-dir", type=Path, default=Path("data/test_ids"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-size", type=float, default=VAL_SIZE_FROM_REMAINING)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_specs(test_ids_dir=args.test_ids_dir, data_dir=args.data_dir)
    datasets = list(specs) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        prepare_dataset(specs[dataset], val_size=args.val_size)


if __name__ == "__main__":
    main()

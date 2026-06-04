"""
Prepare all supported datasets with fixed official test identifiers.

Expected input files:
    data/test_ids/gender_test_ids.csv
    data/test_ids/age_test_ids.csv
    data/test_ids/rosbank_test_ids.csv

Outputs for each dataset:
    data/<dataset>/train.csv
    data/<dataset>/val.csv
    data/<dataset>/test.csv
    data/<dataset>/split_report.json
    data/<dataset>/split_summary.txt

Examples:
    python prepare_official_splits.py --dataset all
    python prepare_official_splits.py --dataset gender --test-ids-dir data/test_ids
    python prepare_official_splits.py --dataset age --val-size-from-remaining 0.111111
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.official_splits import DEFAULT_VAL_SIZE_FROM_REMAINING, build_specs, prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed official test-id splits.")
    parser.add_argument(
        "--dataset",
        choices=["all", "gender", "age", "rosbank"],
        default="all",
        help="Dataset to prepare.",
    )
    parser.add_argument(
        "--test-ids-dir",
        type=Path,
        default=Path("data/test_ids"),
        help="Directory containing *_test_ids.csv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root directory for prepared data/<dataset> outputs.",
    )
    parser.add_argument(
        "--val-size-from-remaining",
        type=float,
        default=DEFAULT_VAL_SIZE_FROM_REMAINING,
        help=(
            "Validation fraction among non-test labeled clients. "
            "The default keeps approximately 80/10/10 train/val/test when official test is 10%."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_specs(test_ids_dir=args.test_ids_dir, output_root=args.output_root)

    selected = list(specs) if args.dataset == "all" else [args.dataset]
    for name in selected:
        prepare_dataset(specs[name], val_size_from_remaining=args.val_size_from_remaining)


if __name__ == "__main__":
    main()

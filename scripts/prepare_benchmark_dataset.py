#!/usr/bin/env python3
"""Prepare immutable fold manifests for Data Fusion Education or Berka."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.benchmark_registry import BENCHMARKS, prepare_benchmark


KAGGLE_DATASET = "konstantinalbul/vtbdatafusion2022"


def download_datafusion(raw_root: Path) -> None:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "The Kaggle CLI is required for --download. Configure KAGGLE_USERNAME "
            "and KAGGLE_KEY, or place transactions.csv and train.csv manually."
        )
    raw_root.mkdir(parents=True, exist_ok=True)
    archive = raw_root / "vtbdatafusion2022.zip"
    subprocess.run(
        [
            executable, "datasets", "download", "-d", KAGGLE_DATASET,
            "-p", str(raw_root), "--force",
        ],
        check=True,
    )
    candidates = sorted(raw_root.glob("*.zip"))
    if not candidates:
        raise FileNotFoundError("Kaggle download produced no ZIP archive")
    archive = candidates[0]
    with zipfile.ZipFile(archive) as bundle:
        wanted = {
            name for name in bundle.namelist()
            if Path(name).name in {"transactions.csv", "train.csv"}
        }
        if len(wanted) != 2:
            raise RuntimeError(
                "Kaggle archive does not contain transactions.csv and train.csv"
            )
        bundle.extractall(raw_root, members=sorted(wanted))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/benchmarks_v5")
    )
    parser.add_argument(
        "--split-backend",
        choices=("sklearn_public", "pyspark", "sklearn_approx"),
        default="sklearn_public",
        help=("Data Fusion only; sklearn_public exactly reproduces the public "
              "KFold(n_splits=5, shuffle=True, random_state=100) protocol."),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    raw_root = args.raw_root or (
        Path("data/external/berka/raw")
        if args.dataset == "berka"
        else Path("data/external/data_fusion_education/raw")
    )
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "dataset": args.dataset,
        "protocol": BENCHMARKS[args.dataset].protocol,
        "raw_root": str(raw_root),
        "output_root": str(args.output_root),
        "download": bool(args.download),
        "split_backend": (
            args.split_backend
            if args.dataset == "datafusion_education"
            else "numpy_random_state_protocol_match"
        ),
        "modality": "transactions_only",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if args.download:
        if args.dataset != "datafusion_education":
            raise ValueError(
                "Berka is already stored under data/external/berka/raw; "
                "--download is only implemented for Data Fusion."
            )
        download_datafusion(raw_root)
    result = prepare_benchmark(
        args.dataset,
        raw_root=raw_root,
        output_root=args.output_root,
        split_backend=args.split_backend,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

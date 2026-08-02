#!/usr/bin/env python3
"""Download and prepare an isolated benchmark with strict provenance checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks import cofinfad, datafusion_default_2023


ADAPTERS = {
    datafusion_default_2023.NAME: datafusion_default_2023,
    cofinfad.NAME: cofinfad,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/isolated_benchmarks")
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    adapter = ADAPTERS[args.dataset]
    raw_root = args.raw_root / args.dataset
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "dataset": args.dataset,
        "raw_root": str(raw_root),
        "output_root": str(args.output_root),
        "downloads": {
            name: {"url": value[0], "sha256": value[1]}
            for name, value in adapter.FILES.items()
        },
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.execute:
        return
    manifest = adapter.prepare(raw_root, args.output_root)
    print(json.dumps({"manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()

"""Dry-run-first matrix for LLM, clustering, model, and ML stability."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

SEEDS = [17, 101, 947]
THRESHOLDS = [0.008, 0.010, 0.012]
CLUSTER_COUNTS = [100, 200, 400, 800]
COVERAGE = [2, 5, 10]
ENCODINGS = ["binary", "raw_count", "normalized_count"]


def build_plan(run_id, datasets):
    cells = []
    for dataset in datasets:
        cells.extend([
            {"dataset": dataset, "axis": "generation_seed", "values": SEEDS, "clients": 300, "api": True},
            {"dataset": dataset, "axis": "clustering_order_seed", "values": SEEDS, "scope": "full train claims", "api": False},
            {"dataset": dataset, "axis": "distance_threshold", "values": THRESHOLDS, "api": False},
            {"dataset": dataset, "axis": "fixed_cluster_count", "values": CLUSTER_COUNTS, "api": False},
            {"dataset": dataset, "axis": "min_client_coverage", "values": COVERAGE, "api": False},
            {"dataset": dataset, "axis": "feature_encoding", "values": ENCODINGS, "api": False},
            {"dataset": dataset, "axis": "ml_seed", "values": SEEDS, "after": "validation tuning frozen", "api": False},
        ])
    return {
        "run_id": run_id, "seeds": SEEDS, "cells": cells,
        "reporting": ["claim variation", "medoid matching", "anchor ARI/NMI", "downstream mean±SD", "paired client bootstrap CI"],
        "rule": "never pool seeds and granularity into one mean",
        "cross_model": {"models": ["qwen", "gpt_oss"], "metrics": ["one-to-one", "mutual nearest", "weighted cosine", "coverage", "unmatched mass"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v2")
    parser.add_argument("--datasets", nargs="+", default=["gender", "age", "rosbank"])
    parser.add_argument("--output", type=Path, default=Path("logs/runs/reviewer-v2/robustness.queue.json"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.run_id, args.datasets)
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", **plan}, indent=2))
    if not args.execute:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.execute_api and not args.until_complete:
        raise ValueError("--execute-api requires --until-complete")
    print(f"Materialized robustness queue -> {args.output}")


if __name__ == "__main__":
    main()

"""Materialize and execute offline robustness cells; dry-run by default."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config_builder import deep_merge, load_yaml, write_runtime_config

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
        "run_id": run_id,
        "seeds": SEEDS,
        "cells": cells,
        "reporting": [
            "claim variation",
            "medoid matching",
            "anchor ARI/NMI",
            "downstream mean±SD",
            "client bootstrap CI",
        ],
        "rule": "never pool seeds and granularity into one mean",
        "cross_model": {
            "models": ["qwen", "gpt_oss"],
            "metrics": [
                "one-to-one",
                "mutual nearest",
                "weighted cosine",
                "coverage",
                "unmatched mass",
            ],
        },
        "api_execution": {
            "supported": False,
            "reason": "generation-seed subset API queue is not implemented",
        },
    }


def materialize_offline_config(
    *,
    dataset: str,
    run_id: str,
    generated_dir: Path,
    results_root: Path,
    legacy_results_root: Path,
) -> Path:
    base = load_yaml(Path("configs") / f"{dataset}.yaml")
    output_dir = results_root / dataset / "legacy_offline" / "offline" / "seed_17"
    overlay: dict[str, Any] = {
        "experiment": {
            "run_id": run_id,
            "variant": "legacy_offline",
            "model_slug": "offline",
            "seed": 17,
        },
        "evaluation": {
            "seeds": SEEDS,
            "bootstrap_samples": 1000,
        },
        "clustering": {
            "mode": "label_agnostic",
            "feature_encoding": "binary",
            "min_client_coverage": 5,
        },
        "input": {
            "cot_features_base_dir": str(legacy_results_root / dataset),
            "claims_base_dir": str(legacy_results_root / dataset),
        },
    }
    config = deep_merge(base, overlay)
    config["output"]["base_dir"] = str(output_dir)
    path = generated_dir / f"{dataset}_offline.yaml"
    return write_runtime_config(path, config)


def commands_for(
    config_path: Path,
    *,
    dataset: str,
    legacy_results_root: Path,
) -> list[list[str]]:
    config = load_yaml(config_path)
    output_dir = config["output"]["base_dir"]
    return [
        [
            sys.executable,
            "run_pipeline.py",
            "--config",
            str(config_path),
            "--steps",
            "stats,ml",
            "--experiments",
            "standard,handcrafted,cot,concat",
            "--splits",
            "train,val,test",
            "--execute",
        ],
        [
            sys.executable,
            "scripts/cluster_stability_experiment.py",
            "--config",
            str(config_path),
            "--claims-base-dir",
            str(legacy_results_root / dataset),
            "--output-base-dir",
            str(output_dir),
            "--seeds",
            *[str(seed) for seed in SEEDS],
            "--distance-thresholds",
            *[str(value) for value in THRESHOLDS],
            "--n-clusters",
            *[str(value) for value in CLUSTER_COUNTS],
            "--coverage-thresholds",
            *[str(value) for value in COVERAGE],
            "--feature-encodings",
            *ENCODINGS,
            "--execute",
        ],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v2")
    parser.add_argument("--datasets", nargs="+", default=["gender", "age", "rosbank"])
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/v2"))
    parser.add_argument("--legacy-results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args.run_id, args.datasets)
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", **plan}, indent=2))
    if not args.execute:
        return
    if args.execute_api and not args.until_complete:
        raise ValueError("--execute-api requires --until-complete")
    if args.execute_api:
        raise RuntimeError(
            "Generation-seed API execution is not implemented; no files or "
            "offline jobs were created. Run with --execute only for offline cells."
        )

    generated_dir = args.generated_dir or Path("logs/runs") / args.run_id / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    queue = []
    for dataset in args.datasets:
        config_path = materialize_offline_config(
            dataset=dataset,
            run_id=args.run_id,
            generated_dir=generated_dir,
            results_root=args.results_root,
            legacy_results_root=args.legacy_results_root,
        )
        for command in commands_for(
            config_path,
            dataset=dataset,
            legacy_results_root=args.legacy_results_root,
        ):
            queue.append({"dataset": dataset, "command": command})

    output = args.output or generated_dir / "robustness.queue.json"
    write_runtime_config(output.with_suffix(".yaml"), {"jobs": queue, "plan": plan})
    output.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    for job in queue:
        subprocess.run(job["command"], check=True)
    print(f"Completed offline robustness queue -> {output}")


if __name__ == "__main__":
    main()

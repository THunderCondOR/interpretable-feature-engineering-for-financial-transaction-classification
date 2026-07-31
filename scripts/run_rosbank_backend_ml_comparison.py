#!/usr/bin/env python3
"""Evaluate Rosbank CoT/Concat ML over MiniBatch clustering seeds."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v4_offline_pipeline import load_source
from src.experiments.artifacts import atomic_write_json
from src.models.ml_baseline import run_ml_baseline


MODELS = ("qwen", "gpt_oss")
EXPERIMENTS = ("cot", "concat")


def metric_summary(metrics: dict, feature_set: str) -> dict:
    summary = metrics[feature_set]["xgboost"]["summary"]["test"]
    return {
        metric: summary[metric]
        for metric in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc")
        if metric in summary
    }


def run_cell(seed_root: Path, source_root: Path) -> dict:
    _, config = load_source(source_root)
    derived = copy.deepcopy(config)
    derived["output"]["base_dir"] = str(seed_root)
    derived.setdefault("input", {})["cot_features_base_dir"] = str(seed_root)
    derived.setdefault("evaluation", {})["seeds"] = [17, 101, 947]
    derived["evaluation"]["bootstrap_samples"] = 1000
    derived.setdefault("optuna", {})["n_trials"] = 30
    derived.setdefault("experiment", {})["run_id"] = (
        "reviewer-v4-rosbank-minibatch-comparison"
    )
    run_ml_baseline(derived, experiments=list(EXPERIMENTS))
    metrics = json.loads(
        (seed_root / "ml_metrics.json").read_text(encoding="utf-8")
    )
    return {
        feature_set: metric_summary(metrics, feature_set)
        for feature_set in EXPERIMENTS
    }


def comparison_rows(derived_root: Path, results: list[dict]) -> list[dict]:
    rows = []
    for model in MODELS:
        legacy = json.loads(
            (
                derived_root / "rosbank" / model / "seed_17"
                / "ml_metrics.json"
            ).read_text(encoding="utf-8")
        )
        model_results = [row for row in results if row["model"] == model]
        for feature_set in EXPERIMENTS:
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "roc_auc",
            ):
                values = [
                    float(row["metrics"][feature_set][metric]["mean"])
                    for row in model_results
                ]
                legacy_value = float(
                    legacy[feature_set]["xgboost"]["summary"]["test"][metric][
                        "mean"
                    ]
                )
                minibatch_mean = statistics.mean(values)
                rows.append({
                    "model": model,
                    "feature_set": feature_set,
                    "metric": metric,
                    "legacy_agglomerative": legacy_value,
                    "minibatch_cluster_seed_mean": minibatch_mean,
                    "minibatch_cluster_seed_sd": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                    "minibatch_min": min(values),
                    "minibatch_max": max(values),
                    "delta": minibatch_mean - legacy_value,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("results/v2/derived/reviewer-v4-offline-v2"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )
    parser.add_argument("--cluster-seeds", nargs="+", type=int, default=[17, 101, 947])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild legacy deltas from an already completed result file.",
    )
    args = parser.parse_args()

    jobs = []
    for model in args.models:
        cell = args.derived_root / "rosbank" / model / "seed_17"
        source = json.loads(
            (cell / "source_manifest.json").read_text(encoding="utf-8")
        )["source_contract"]
        for seed in args.cluster_seeds:
            seed_root = (
                cell / "stability" / "cluster_seeds_minibatch_kmeans"
                / f"seed_{seed}"
            )
            jobs.append({
                "model": model,
                "cluster_seed": seed,
                "seed_root": str(seed_root),
                "source_root": source["source_root"],
            })
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "experiments": list(EXPERIMENTS),
        "classifier_seeds": [17, 101, 947],
        "jobs": jobs,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    comparison_path = (
        args.derived_root / "rosbank_minibatch_ml_comparison.json"
    )
    if args.summarize_only:
        results = []
        for job in jobs:
            metrics_path = Path(job["seed_root"]) / "ml_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            results.append({
                **job,
                "metrics": {
                    feature_set: metric_summary(metrics, feature_set)
                    for feature_set in EXPERIMENTS
                },
            })
        atomic_write_json(comparison_path, {
            "protocol": plan,
            "results": results,
            "comparison": comparison_rows(args.derived_root, results),
            "state": "completed",
            "summarized_at": datetime.now(timezone.utc).isoformat(),
        })
        return

    results = []
    for job in jobs:
        seed_root = Path(job["seed_root"])
        required = [
            seed_root / f"cot_features_{split}.parquet"
            for split in ("train", "val", "test")
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "MiniBatch features are incomplete; rerun cluster stability: "
                + ", ".join(missing)
            )
        metrics = run_cell(seed_root, Path(job["source_root"]))
        results.append({**job, "metrics": metrics})
        atomic_write_json(
            comparison_path,
            {
                "protocol": plan,
                "results": results,
                "state": "running",
            },
        )
    atomic_write_json(
        comparison_path,
        {
            "protocol": plan,
            "results": results,
            "comparison": comparison_rows(args.derived_root, results),
            "state": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()

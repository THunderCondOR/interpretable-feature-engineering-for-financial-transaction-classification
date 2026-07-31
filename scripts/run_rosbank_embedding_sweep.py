#!/usr/bin/env python3
"""Run a resumable Rosbank embedding-model sweep on immutable claims."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json


EMBEDDERS = {
    "gte_multilingual_base": "Alibaba-NLP/gte-multilingual-base",
    "multilingual_e5_large": "intfloat/multilingual-e5-large",
    "bge_m3": "BAAI/bge-m3",
}
SOURCES = {
    "qwen": Path("results/v2/rosbank/guided_zero_shot_v4/qwen/seed_17"),
    "gpt_oss": Path(
        "results/v2/rosbank/guided_zero_shot_v4/gpt_oss/seed_17"
    ),
}


def metric_means(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return {
        feature_set: {
            metric: metrics[feature_set]["xgboost"]["summary"]["test"][
                metric
            ]["mean"]
            for metric in ("accuracy", "balanced_accuracy", "roc_auc")
        }
        for feature_set in ("cot", "concat")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/v2/derived/rosbank-embedding-sweep-v1"),
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        choices=tuple(EMBEDDERS),
        default=list(EMBEDDERS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(SOURCES),
        default=list(SOURCES),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    jobs = [
        {
            "embedder_slug": slug,
            "embedding_model": EMBEDDERS[slug],
            "source_model": model,
            "source_root": str(SOURCES[model]),
            "derived_root": str(args.output_root / slug),
        }
        for slug in args.embedders
        for model in args.models
    ]
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "dataset": "rosbank",
        "clustering": "minibatch_kmeans",
        "candidates": ["k_200", "k_400", "k_800"],
        "selection_split": "validation",
        "ml_experiments": ["cot", "concat"],
        "jobs": jobs,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    state_path = args.output_root / "embedding_sweep.json"
    state = {
        **plan,
        "state": "running",
        "completed": [],
        "failed": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(state_path, state)
    for job in jobs:
        key = f"{job['embedder_slug']}:{job['source_model']}"
        state["current"] = key
        atomic_write_json(state_path, state)
        command = [
            sys.executable,
            "scripts/run_v4_offline_pipeline.py",
            "--source-root",
            job["source_root"],
            "--derived-root",
            job["derived_root"],
            "--embedding-model",
            job["embedding_model"],
            "--clustering-backend",
            "minibatch_kmeans",
            "--candidates",
            "k_200",
            "k_400",
            "k_800",
            "--ml-experiments",
            "cot",
            "concat",
            "--execute",
        ]
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            cell = (
                Path(job["derived_root"]) / "rosbank"
                / job["source_model"] / "seed_17"
            )
            selection = json.loads(
                (cell / "cluster_selection.json").read_text(encoding="utf-8")
            )
            state["completed"].append({
                "key": key,
                "cell": str(cell),
                "selected_candidate": selection["selected_candidate"],
                "selected_representation": selection[
                    "selected_representation"
                ],
                "test_metrics": metric_means(cell / "ml_metrics.json"),
            })
        except Exception as error:
            state["failed"].append({
                "key": key,
                "type": type(error).__name__,
                "message": str(error),
            })
        atomic_write_json(state_path, state)
    state["state"] = (
        "completed_with_errors" if state["failed"] else "completed"
    )
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()

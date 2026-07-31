#!/usr/bin/env python3
"""Select a scalable clustering family on Rosbank E5 claim embeddings."""
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

from src.experiments.artifacts import atomic_write_json, fingerprint


SOURCES = {
    "qwen": Path("results/v2/rosbank/guided_zero_shot_v4/qwen/seed_17"),
    "gpt_oss": Path(
        "results/v2/rosbank/guided_zero_shot_v4/gpt_oss/seed_17"
    ),
}
E5_CACHE_ROOT = Path(
    "results/v2/derived/rosbank-embedding-sweep-v1/"
    "multilingual_e5_large/rosbank"
)
BACKENDS = {
    "minibatch_kmeans": ["k_100", "k_200", "k_400", "k_800"],
    "spherical_kmeans": ["k_100", "k_200", "k_400", "k_800"],
    "hdbscan_pca": [
        f"pca{dimensions}_mcs{minimum}_ms{samples}"
        for dimensions in (64, 128)
        for minimum in (10, 25, 50)
        for samples in (5, 10)
    ],
    "agglomerative": ["k_100", "k_200", "k_400", "k_800"],
}
SEEDS = (17, 101, 947)
SIMPLICITY = {
    "minibatch_kmeans": 0,
    "spherical_kmeans": 1,
    "hdbscan_pca": 2,
}


def jobs(output_root: Path) -> list[dict]:
    rows = []
    for backend, candidates in BACKENDS.items():
        seeds = (17,) if backend == "agglomerative" else SEEDS
        for model, source in SOURCES.items():
            for seed in seeds:
                rows.append({
                    "backend": backend,
                    "model": model,
                    "seed": seed,
                    "source_root": str(source),
                    "embedding_cache_cell": str(
                        E5_CACHE_ROOT / model / "seed_17"
                    ),
                    "derived_root": str(
                        output_root / backend / f"cluster_seed_{seed}"
                    ),
                    "candidates": candidates,
                })
    return rows


def summarize(completed: list[dict]) -> tuple[dict, dict]:
    scalable = {}
    for backend in SIMPLICITY:
        cells = [row for row in completed if row["backend"] == backend]
        per_model = {}
        for model in SOURCES:
            values = [
                float(row["validation_balanced_accuracy"])
                for row in cells if row["model"] == model
            ]
            if len(values) != len(SEEDS):
                continue
            per_model[model] = {
                "mean": sum(values) / len(values),
                "values": values,
            }
        if set(per_model) == set(SOURCES):
            scalable[backend] = {
                "per_model": per_model,
                "mean_validation_balanced_accuracy": sum(
                    value["mean"] for value in per_model.values()
                ) / len(per_model),
            }
    if not scalable:
        raise RuntimeError("No scalable clustering family completed all cells")
    best = max(
        value["mean_validation_balanced_accuracy"]
        for value in scalable.values()
    )
    eligible = [
        backend for backend, value in scalable.items()
        if best - value["mean_validation_balanced_accuracy"] <= 0.005
    ]
    winner = min(eligible, key=lambda backend: SIMPLICITY[backend])
    selection = {
        "selection_split": "validation",
        "primary_metric": "mean CoT balanced accuracy across Qwen/GPT-OSS",
        "tie_margin": 0.005,
        "scalable_families": scalable,
        "selected_backend": winner,
    }
    selection["selection_signature"] = fingerprint(selection)
    return selection, scalable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/v2/derived/rosbank-clustering-sweep-e5-v1"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip the non-scalable agglomerative reference in production queues.",
    )
    args = parser.parse_args()
    queue = jobs(args.output_root)
    if args.skip_reference:
        queue = [job for job in queue if job["backend"] != "agglomerative"]
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "embedding_model": "intfloat/multilingual-e5-large",
        "selection_split": "validation",
        "agglomerative_reference": not args.skip_reference,
        "jobs": queue,
    }
    plan["plan_signature"] = fingerprint({
        key: value for key, value in plan.items() if key != "mode"
    })
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return

    state_path = args.output_root / "clustering_sweep.json"
    previous_completed = []
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("plan_signature") == plan["plan_signature"]:
            previous_completed = list(previous.get("completed", []))
    state = {
        **plan,
        "state": "running",
        "completed": previous_completed,
        "failed": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    completed_keys = {row["key"] for row in state["completed"]}
    atomic_write_json(state_path, state)
    for job in queue:
        key = f"{job['backend']}:{job['model']}:seed_{job['seed']}"
        if key in completed_keys:
            continue
        state["current"] = key
        atomic_write_json(state_path, state)
        scalable_done = all(
            any(
                row["backend"] == backend
                and row["model"] == model
                and int(row["seed"]) == seed
                for row in state["completed"]
            )
            for backend in SIMPLICITY
            for model in SOURCES
            for seed in SEEDS
        )
        if scalable_done and not (args.output_root / "selection.json").is_file():
            interim_selection, _ = summarize(state["completed"])
            atomic_write_json(
                args.output_root / "selection.json", interim_selection
            )
        command = [
            sys.executable,
            "scripts/run_v4_offline_pipeline.py",
            "--source-root", job["source_root"],
            "--derived-root", job["derived_root"],
            "--embedding-model", "intfloat/multilingual-e5-large",
            "--embedding-cache-cell", job["embedding_cache_cell"],
            "--clustering-backend", job["backend"],
            "--clustering-seed", str(job["seed"]),
            "--candidates", *job["candidates"],
            "--skip-ml",
            "--execute",
        ]
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            cell = (
                Path(job["derived_root"]) / "rosbank"
                / job["model"] / "seed_17"
            )
            selection = json.loads(
                (cell / "cluster_selection.json").read_text(encoding="utf-8")
            )
            state["completed"].append({
                "key": key,
                "backend": job["backend"],
                "model": job["model"],
                "seed": job["seed"],
                "cell": str(cell),
                "selected_candidate": selection["selected_candidate"],
                "selected_representation": selection[
                    "selected_representation"
                ],
                "validation_balanced_accuracy": selection[
                    "selected_representation"
                ]["validation_balanced_accuracy"],
            })
            completed_keys.add(key)
        except Exception as error:
            state["failed"].append({
                "key": key,
                "type": type(error).__name__,
                "message": str(error),
            })
        atomic_write_json(state_path, state)

    try:
        selection, _ = summarize(state["completed"])
        atomic_write_json(args.output_root / "selection.json", selection)
        state["selection"] = selection
    except Exception as error:
        state["selection_error"] = str(error)
    state["state"] = (
        "completed_with_errors" if state["failed"] else "completed"
    )
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()

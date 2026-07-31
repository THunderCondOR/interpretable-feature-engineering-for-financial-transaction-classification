#!/usr/bin/env python3
"""Rebuild all v4 datasets with the Rosbank-selected E5 clustering family."""
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


DATASETS = ("rosbank", "gender", "age")
MODELS = ("qwen", "gpt_oss")
SOURCE_VARIANTS = {
    "rosbank": "guided_zero_shot_v4",
    "gender": "guided_zero_shot_v4",
    # Age v4 deliberately used opaque ordinal proxy labels.  Its artifact
    # directory includes the semantic variant suffix and is immutable.
    "age": "guided_zero_shot_v4__age_opaque",
}
E5_CACHE = Path(
    "results/v2/derived/rosbank-embedding-sweep-v1/"
    "multilingual_e5_large/rosbank/{model}/seed_17"
)
BACKEND_CANDIDATES = {
    "minibatch_kmeans": ("k_100", "k_200", "k_400", "k_800"),
    "spherical_kmeans": ("k_100", "k_200", "k_400", "k_800"),
    "hdbscan_pca": tuple(
        f"pca{dimension}_mcs{minimum}_ms{samples}"
        for dimension in (64, 128)
        for minimum in (10, 25, 50)
        for samples in (5, 10)
    ),
}
ML_EXPERIMENTS = (
    "standard", "llm_profile", "standard_profile", "handcrafted",
    "all_nonclaim", "cot", "concat", "standard_cot", "all_features",
)


def build_jobs(
    output_root: Path,
    selected_backend: str,
) -> list[dict[str, object]]:
    if selected_backend not in BACKEND_CANDIDATES:
        raise ValueError(f"Unsupported selected backend: {selected_backend}")
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            source = (
                Path("results/v2") / dataset / SOURCE_VARIANTS[dataset]
                / model / "seed_17"
            )
            row: dict[str, object] = {
                "dataset": dataset,
                "model": model,
                "source_root": str(source),
                "derived_root": str(output_root),
                "backend": selected_backend,
                "candidates": list(BACKEND_CANDIDATES[selected_backend]),
            }
            if dataset == "rosbank":
                row["embedding_cache_cell"] = str(
                    Path(str(E5_CACHE).format(model=model))
                )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "results/v2/derived/rosbank-clustering-sweep-e5-v1/selection.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/v2/derived/reviewer-v6-e5-clustering"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.selection.is_file():
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        selected_backend = str(selection["selected_backend"])
    elif args.execute:
        raise FileNotFoundError(f"Missing clustering selection: {args.selection}")
    else:
        selected_backend = "<from-selection>"
        selection = {}

    jobs = (
        build_jobs(args.output_root, selected_backend)
        if selected_backend in BACKEND_CANDIDATES else []
    )
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "selection": str(args.selection),
        "selected_backend": selected_backend,
        "embedding_model": "intfloat/multilingual-e5-large",
        "jobs": jobs or "materialized after clustering selection",
        "ml_experiments": ML_EXPERIMENTS,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return

    missing = [
        job["source_root"] for job in jobs
        if not (REPO_ROOT / str(job["source_root"]) / "manifest.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing immutable API sources: {missing}")

    state_path = args.output_root / "rebuild_state.json"
    completed: list[dict[str, str]] = []
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("selection_signature") == selection.get(
            "selection_signature"
        ):
            completed = list(previous.get("completed", []))
    completed_keys = {row["key"] for row in completed}
    state = {
        **plan,
        "state": "running",
        "completed": completed,
        "selection_signature": selection.get("selection_signature"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(state_path, state)
    for job in jobs:
        key = f"{job['dataset']}:{job['model']}"
        if key in completed_keys:
            continue
        state["current"] = key
        atomic_write_json(state_path, state)
        command = [
            sys.executable, "scripts/run_v4_offline_pipeline.py",
            "--source-root", str(job["source_root"]),
            "--derived-root", str(job["derived_root"]),
            "--embedding-model", "intfloat/multilingual-e5-large",
            "--clustering-backend", str(job["backend"]),
            "--clustering-seed", "17",
            "--candidates", *map(str, job["candidates"]),
            "--ml-experiments", *ML_EXPERIMENTS,
            "--execute",
        ]
        if job.get("embedding_cache_cell"):
            command[command.index("--clustering-backend"):command.index(
                "--clustering-backend"
            )] = [
                "--embedding-cache-cell", str(job["embedding_cache_cell"])
            ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        state["completed"].append({"key": key, "state": "completed"})
        completed_keys.add(key)
        atomic_write_json(state_path, state)

    state["state"] = "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()

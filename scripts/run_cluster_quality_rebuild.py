#!/usr/bin/env python3
"""Rebuild all core datasets with validation-selected cluster-quality settings."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cluster_quality_sweep import (
    CANDIDATES,
    MODELS,
    SOURCE_VARIANTS,
    cache_cell,
    source_root,
)
from src.experiments.artifacts import atomic_write_json


ML_EXPERIMENTS = (
    "standard", "llm_profile", "standard_profile", "handcrafted",
    "all_nonclaim", "cot", "concat", "standard_cot", "all_features",
)


def selected_rosbank_sweep_root(
    selection_path: Path,
    selection: dict,
    model: str,
) -> Path:
    """Locate the already-materialized winning Rosbank sweep root."""
    selected = selection["selected_configuration"]
    geometry = str(selected["geometry"])
    quantile = selected.get("assignment_quantile")
    coverage = int(selected["min_client_coverage"])
    quantile_slug = (
        "global" if quantile is None
        else f"q{int(float(quantile) * 100)}"
    )
    return (
        selection_path.parent / "assignment" / geometry
        / f"{quantile_slug}_coverage_{coverage}" / model
    )


def expose_cell(source_cell: Path, destination_cell: Path) -> None:
    """Expose a sweep cell without copying or recomputing its clusters."""
    if destination_cell.is_symlink():
        if destination_cell.resolve() != source_cell.resolve():
            raise ValueError(
                f"Incompatible existing cell symlink: {destination_cell}"
            )
        return
    if destination_cell.exists():
        raise FileExistsError(
            f"Refusing to replace existing rebuild cell: {destination_cell}"
        )
    destination_cell.parent.mkdir(parents=True, exist_ok=True)
    destination_cell.symlink_to(source_cell.resolve(), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path,
        default=Path(
            "results/v2/derived/cluster-quality-v2/rosbank/selection.json"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(
            "results/v2/derived/reviewer-v7-cluster-quality"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    selection = (
        json.loads(args.selection.read_text(encoding="utf-8"))
        if args.selection.is_file() else {}
    )
    if args.execute and not selection:
        raise FileNotFoundError(args.selection)
    selected = selection.get("selected_configuration", {})
    geometry = selected.get("geometry", "<from-selection>")
    quantile = selected.get("assignment_quantile")
    coverage = selected.get("min_client_coverage", "<from-selection>")
    jobs = [
        {"dataset": dataset, "model": model}
        for dataset in SOURCE_VARIANTS for model in MODELS
    ]
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "selection": str(args.selection),
        "output_root": str(args.output_root),
        "embedding_model": "intfloat/multilingual-e5-large",
        "geometry": geometry,
        "assignment_quantile": quantile,
        "min_client_coverage": coverage,
        "cluster_candidates": CANDIDATES,
        "jobs": jobs,
        "ml_experiments": ML_EXPERIMENTS,
        "rosbank_strategy": (
            "reuse winning sweep cluster artifacts; run ML only"
        ),
        "gender_age_strategy": "materialize selected settings once",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return

    state_path = args.output_root / "rebuild_state.json"
    previous = {}
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
    completed = set(previous.get("completed", []))
    state = {**plan, "state": "running", "completed": sorted(completed)}
    atomic_write_json(state_path, state)
    for job in jobs:
        key = f"{job['dataset']}:{job['model']}"
        cell = (
            args.output_root / job["dataset"] / job["model"] / "seed_17"
        )
        if key in completed and (cell / "ml_metrics.json").is_file():
            continue
        derived_root = (
            args.output_root
            if job["dataset"] != "rosbank"
            else selected_rosbank_sweep_root(
                args.selection, selection, job["model"]
            )
        )
        command = [
            sys.executable,
            "scripts/run_v4_offline_pipeline.py",
            "--source-root", str(
                source_root(job["dataset"], job["model"])
            ),
            "--derived-root", str(derived_root),
            "--embedding-model", "intfloat/multilingual-e5-large",
            "--embedding-transform", str(geometry),
            "--clustering-backend", "minibatch_kmeans",
            "--clustering-seed", "17",
            "--min-client-coverage", str(coverage),
            "--candidates", *CANDIDATES,
            "--ml-experiments", *ML_EXPERIMENTS,
            "--execute",
        ]
        if quantile is not None:
            command.extend(["--assignment-quantile", str(quantile)])
        if job["dataset"] == "rosbank":
            command.extend([
                "--embedding-cache-cell", str(cache_cell(job["model"])),
            ])
        state["current"] = key
        state["current_started_at_epoch"] = time.time()
        atomic_write_json(state_path, state)
        print(
            json.dumps({"event": "rebuild_started", "cell": key}),
            flush=True,
        )
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            if job["dataset"] == "rosbank":
                source_cell = (
                    derived_root / "rosbank" / job["model"] / "seed_17"
                )
                if not (source_cell / "ml_metrics.json").is_file():
                    raise FileNotFoundError(source_cell / "ml_metrics.json")
                expose_cell(source_cell, cell)
        except Exception as error:
            state["state"] = "failed"
            state["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(state_path, state)
            raise
        completed.add(key)
        state["completed"] = sorted(completed)
        state["last_completed"] = key
        state["last_elapsed_seconds"] = round(
            time.time() - state["current_started_at_epoch"], 3
        )
        atomic_write_json(state_path, state)
        print(
            json.dumps({
                "event": "rebuild_completed",
                "cell": key,
                "elapsed_seconds": state["last_elapsed_seconds"],
            }),
            flush=True,
        )
    state["state"] = "completed"
    state["current"] = None
    state.pop("current_started_at_epoch", None)
    state.pop("error", None)
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tune cached E5 cluster geometry and noise filtering without API calls."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json, fingerprint


MODELS = ("qwen", "gpt_oss")
SOURCE_VARIANTS = {
    "rosbank": "guided_zero_shot_v4",
    "gender": "guided_zero_shot_v4",
    "age": "guided_zero_shot_v4__age_opaque",
}
GEOMETRIES = ("raw", "centered", "pca_whiten_64", "pca_whiten_128")
ASSIGNMENT_QUANTILES = (None, 0.90, 0.95, 0.99)
COVERAGES = (2, 5, 10, 20)
CANDIDATES = ("k_50", "k_100", "k_150", "k_200", "k_300", "k_400", "k_600")


def source_root(dataset: str, model: str) -> Path:
    return (
        Path("results/v2") / dataset / SOURCE_VARIANTS[dataset]
        / model / "seed_17"
    )


def cache_cell(model: str) -> Path:
    return (
        Path("results/v2/derived/rosbank-embedding-sweep-v1")
        / "multilingual_e5_large" / "rosbank" / model / "seed_17"
    )


def cell_root(derived_root: Path, dataset: str, model: str) -> Path:
    return derived_root / dataset / model / "seed_17"


def run_job(job: dict, *, execute: bool) -> dict:
    started = time.monotonic()
    derived_root = Path(job["derived_root"])
    cell = cell_root(derived_root, job["dataset"], job["model"])
    selection_path = cell / "cluster_selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        result = {
            **job,
            "state": "reused",
            "validation_balanced_accuracy": float(
                selection["selected_representation"][
                    "validation_balanced_accuracy"
                ]
            ),
            "selected_candidate": selection["selected_candidate"],
            "selected_representation": {
                key: value
                for key, value in selection["selected_representation"].items()
                if key != "selected_feature_names"
            },
        }
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return result
    if not execute:
        return {**job, "state": "planned"}
    command = [
        sys.executable,
        "scripts/run_v4_offline_pipeline.py",
        "--source-root", job["source_root"],
        "--derived-root", job["derived_root"],
        "--embedding-model", "intfloat/multilingual-e5-large",
        "--embedding-transform", job["geometry"],
        "--clustering-backend", "minibatch_kmeans",
        "--clustering-seed", "17",
        "--min-client-coverage", str(job["min_client_coverage"]),
        "--candidates", *CANDIDATES,
        "--skip-ml",
        "--execute",
    ]
    if job["dataset"] == "rosbank":
        command.extend([
            "--embedding-cache-cell", str(cache_cell(job["model"])),
        ])
    if job["assignment_quantile"] is not None:
        command.extend([
            "--assignment-quantile", str(job["assignment_quantile"]),
        ])
    print(
        json.dumps({"event": "job_started", **job}, ensure_ascii=False),
        flush=True,
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = {
        **job,
        "state": "completed",
        "validation_balanced_accuracy": float(
            selection["selected_representation"][
                "validation_balanced_accuracy"
            ]
        ),
        "selected_candidate": selection["selected_candidate"],
        "selected_representation": {
            key: value
            for key, value in selection["selected_representation"].items()
            if key != "selected_feature_names"
        },
    }
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(
        json.dumps(
            {
                "event": "job_completed",
                "model": job["model"],
                "phase": job["phase"],
                "geometry": job["geometry"],
                "assignment_quantile": job["assignment_quantile"],
                "min_client_coverage": job["min_client_coverage"],
                "validation_balanced_accuracy": result[
                    "validation_balanced_accuracy"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for values, cells in grouped.items():
        models = {cell["model"] for cell in cells}
        if models != set(MODELS):
            continue
        result.append({
            **dict(zip(keys, values)),
            "mean_validation_balanced_accuracy": sum(
                cell["validation_balanced_accuracy"] for cell in cells
            ) / len(cells),
            "per_model": {
                cell["model"]: cell["validation_balanced_accuracy"]
                for cell in cells
            },
        })
    return result


def select_with_margin(
    rows: list[dict],
    *,
    tie_margin: float,
    simplicity,
) -> dict:
    if not rows:
        raise RuntimeError("No complete two-model sweep candidates")
    best = max(row["mean_validation_balanced_accuracy"] for row in rows)
    eligible = [
        row for row in rows
        if best - row["mean_validation_balanced_accuracy"] <= tie_margin
    ]
    return sorted(eligible, key=simplicity)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SOURCE_VARIANTS), default="rosbank")
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/v2/derived/cluster-quality-v2"),
    )
    parser.add_argument("--tie-margin", type=float, default=0.005)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    dataset_root = args.output_root / args.dataset
    state_path = dataset_root / "sweep_state.json"
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "dataset": args.dataset,
        "selection_split": "validation",
        "embedding_model": "intfloat/multilingual-e5-large",
        "geometries": GEOMETRIES,
        "assignment_quantiles": ASSIGNMENT_QUANTILES,
        "min_client_coverages": COVERAGES,
        "cluster_candidates": CANDIDATES,
        "tie_margin": args.tie_margin,
    }
    plan["plan_signature"] = fingerprint({
        key: value for key, value in plan.items() if key != "mode"
    })
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    state = {**plan, "state": "geometry", "completed": [], "failed": []}
    atomic_write_json(state_path, state)

    geometry_rows = []
    for geometry in GEOMETRIES:
        for model in MODELS:
            job = {
                "phase": "geometry",
                "dataset": args.dataset,
                "model": model,
                "geometry": geometry,
                "assignment_quantile": None,
                "min_client_coverage": 5,
                "source_root": str(source_root(args.dataset, model)),
                "derived_root": str(
                    dataset_root / "geometry" / geometry / model
                ),
            }
            state["current"] = job
            atomic_write_json(state_path, state)
            try:
                row = run_job(job, execute=True)
                geometry_rows.append(row)
                state["completed"].append(row)
            except Exception as error:
                failure = {
                    **job,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                state["failed"].append(failure)
                print(
                    json.dumps(
                        {"event": "job_failed", **failure},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            atomic_write_json(state_path, state)
    geometry_summary = aggregate(geometry_rows, ("geometry",))
    selected_geometry = select_with_margin(
        geometry_summary,
        tie_margin=args.tie_margin,
        simplicity=lambda row: (
            GEOMETRIES.index(row["geometry"]),
            -row["mean_validation_balanced_accuracy"],
        ),
    )
    state["selected_geometry"] = selected_geometry
    state["state"] = "assignment"
    atomic_write_json(state_path, state)

    assignment_rows = []
    for quantile in ASSIGNMENT_QUANTILES:
        for coverage in COVERAGES:
            for model in MODELS:
                quantile_slug = "global" if quantile is None else f"q{int(quantile * 100)}"
                job = {
                    "phase": "assignment",
                    "dataset": args.dataset,
                    "model": model,
                    "geometry": selected_geometry["geometry"],
                    "assignment_quantile": quantile,
                    "min_client_coverage": coverage,
                    "source_root": str(source_root(args.dataset, model)),
                    "derived_root": str(
                        dataset_root / "assignment"
                        / selected_geometry["geometry"]
                        / f"{quantile_slug}_coverage_{coverage}" / model
                    ),
                }
                state["current"] = job
                atomic_write_json(state_path, state)
                try:
                    row = run_job(job, execute=True)
                    assignment_rows.append(row)
                    state["completed"].append(row)
                except Exception as error:
                    failure = {
                        **job,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    state["failed"].append(failure)
                    print(
                        json.dumps(
                            {"event": "job_failed", **failure},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                atomic_write_json(state_path, state)
    assignment_summary = aggregate(
        assignment_rows,
        ("geometry", "assignment_quantile", "min_client_coverage"),
    )
    selected = select_with_margin(
        assignment_summary,
        tie_margin=args.tie_margin,
        simplicity=lambda row: (
            0 if row["assignment_quantile"] == 0.95 else 1,
            abs(int(row["min_client_coverage"]) - 5),
            -row["mean_validation_balanced_accuracy"],
        ),
    )
    selection = {
        "selection_split": "validation",
        "primary_metric": "mean CoT balanced accuracy across Qwen/GPT-OSS",
        "tie_margin": args.tie_margin,
        "selected_geometry": selected_geometry,
        "selected_configuration": selected,
        "geometry_candidates": geometry_summary,
        "assignment_candidates": assignment_summary,
    }
    selection["selection_signature"] = fingerprint(selection)
    atomic_write_json(dataset_root / "selection.json", selection)
    state["state"] = "completed_with_errors" if state["failed"] else "completed"
    state["current"] = None
    state["selection"] = selection
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()

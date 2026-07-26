#!/usr/bin/env python3
"""Aggregate v5 fold metrics and published transaction-only comparators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.data.benchmark_registry import (
    BENCHMARKS,
    BERKA_BASELINES,
    DATA_FUSION_BASELINES,
)
from src.experiments.artifacts import atomic_write_json
from src.experiments.config_builder import load_yaml


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def direct_rows(
    dataset: str,
    model: str,
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    rows = []
    for fold in range(5):
        selection_path = (
            Path("logs/runs") / run_id / "generated" / dataset
            / f"fold_{fold}" / "prompt_selection.json"
        )
        if not selection_path.is_file():
            continue
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        config = load_yaml(selection["selected_configs"][model]["path"])
        metrics_path = Path(config["output"]["base_dir"]) / "llm_metrics_test.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({
            "dataset": dataset,
            "model": model,
            "experiment": "API LLM direct",
            "classifier": "direct_label",
            "fold": fold,
            "seed": int(config["experiment"]["generation_seed"]),
            "metrics": metrics,
        })
    return rows


def ml_rows(
    dataset: str,
    model: str,
    *,
    derived_root: Path,
) -> list[dict[str, Any]]:
    spec = BENCHMARKS[dataset]
    rows = []
    for fold in range(5):
        path = (
            derived_root / dataset / spec.protocol / f"fold_{fold}" / model
            / "ml_metrics.json"
        )
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for feature_set, feature_payload in payload.items():
            for classifier in ("xgboost", "decision_tree"):
                for seed, run in feature_payload.get(
                    classifier, {}
                ).get("runs", {}).items():
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "experiment": feature_set,
                        "classifier": classifier,
                        "fold": fold,
                        "seed": int(seed),
                        "metrics": run["test"],
                    })
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["dataset"], row["model"], row["experiment"], row["classifier"]
        )
        cells.setdefault(key, []).append(row)
    output = []
    for key, cell in sorted(cells.items()):
        metric_names = sorted({
            metric for row in cell for metric, value in row["metrics"].items()
            if isinstance(value, (int, float))
            and metric not in {"n", "n_rows", "n_scored", "n_skipped", "n_errors"}
        })
        output.append({
            "dataset": key[0],
            "model": key[1],
            "experiment": key[2],
            "classifier": key[3],
            "folds": sorted({row["fold"] for row in cell}),
            "seeds": sorted({row["seed"] for row in cell}),
            "metrics": {
                metric: summary([
                    float(row["metrics"][metric])
                    for row in cell if metric in row["metrics"]
                ])
                for metric in metric_names
            },
        })
    return output


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Fold-based transaction benchmark results",
        "",
        (
            "Data Fusion uses transaction-only MBD-style folds and ROC-AUC. "
            "Berka follows the UniTTab 478/204 repeated-split protocol and "
            "uses positive-class F1; its split IDs are protocol-matched rather "
            "than publication-identical."
        ),
    ]
    for dataset in BENCHMARKS:
        metric = BENCHMARKS[dataset].primary_metric
        lines.extend([
            "",
            f"## {dataset}",
            "",
            f"| Source | Model/features | Classifier | {metric} | N |",
            "|---|---|---|---:|---:|",
        ])
        baselines = payload["published_baselines"][dataset]
        for name, values in baselines.items():
            lines.append(
                f"| Published | {name} | — | "
                f"{values['mean']:.3f} ± {values['sd']:.3f} | 5 folds |"
            )
        for row in payload["results"]:
            if row["dataset"] != dataset or metric not in row["metrics"]:
                continue
            values = row["metrics"][metric]
            lines.append(
                f"| Ours | {row['model']} / {row['experiment']} | "
                f"{row['classifier']} | {values['mean']:.3f} ± "
                f"{values['sd']:.3f} | {values['n']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v5-benchmarks")
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("results/v5/derived/cv_main"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/reviewer-v5-benchmarks"),
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "run_id": args.run_id,
        "datasets": list(BENCHMARKS),
        "models": ["qwen", "gpt_oss"],
        "output_root": str(args.output_root),
    }, indent=2))
    if not args.execute:
        return
    rows = []
    for dataset in BENCHMARKS:
        for model in ("qwen", "gpt_oss"):
            rows.extend(direct_rows(dataset, model, run_id=args.run_id))
            rows.extend(
                ml_rows(dataset, model, derived_root=args.derived_root)
            )
    if not rows and not args.allow_missing:
        raise RuntimeError("No compatible v5 fold results found")
    payload = {
        "run_id": args.run_id,
        "protocols": {
            name: {
                "protocol": spec.protocol,
                "primary_metric": spec.primary_metric,
                "comparison": spec.comparison,
            }
            for name, spec in BENCHMARKS.items()
        },
        "published_baselines": {
            "datafusion_education": DATA_FUSION_BASELINES,
            "berka": BERKA_BASELINES,
        },
        "results": aggregate(rows),
        "incomplete": not bool(rows),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "benchmark_results.json", payload)
    (args.output_root / "benchmark_results.md").write_text(
        markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

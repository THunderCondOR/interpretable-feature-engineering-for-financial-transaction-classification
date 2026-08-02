#!/usr/bin/env python3
"""Validation-only ROC-AUC sweep over frozen Data Fusion cluster candidates."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v4_offline_pipeline import (
    features_from_assignments,
    load_model,
    load_source,
    rank_features_by_train_mi,
)
from src.experiments.artifacts import atomic_write_json
from src.models.ml_baseline import build_feature_sets, merge_feature_frames, split_xy
from src.pipeline.cot_features import load_claim_records


ENCODINGS = ("binary", "raw_count", "normalized_count")
TOP_K = (25, 50, 100, 150, 200)


def auc_fit(
    train, validation, columns: list[str], parameters: dict[str, Any], seed: int = 17,
) -> float:
    x_train, y_train = split_xy(train, columns)
    x_val, y_val = split_xy(validation, columns)
    model = XGBClassifier(
        **parameters,
        objective="binary:logistic",
        random_state=seed,
        n_jobs=2,
        verbosity=0,
        eval_metric="auc",
        tree_method="hist",
    )
    model.fit(x_train, y_train)
    return float(roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]))


def choose(rows: list[dict[str, Any]], tie_margin: float) -> dict[str, Any]:
    best = max(row["validation_concat_roc_auc"] for row in rows)
    eligible = [
        row for row in rows
        if best - row["validation_concat_roc_auc"] <= tie_margin
    ]
    encoding_order = {"binary": 0, "normalized_count": 1, "raw_count": 2}
    return min(
        eligible,
        key=lambda row: (
            row["n_features"], row["n_clusters"],
            encoding_order[row["encoding"]],
            -row["validation_concat_roc_auc"],
        ),
    )


def run_cell(cell: Path, tie_margin: float) -> dict[str, Any]:
    source = json.loads((cell / "source_manifest.json").read_text(encoding="utf-8"))[
        "source_contract"
    ]
    source_root = Path(source["source_root"])
    _, config = load_source(source_root)
    derived = copy.deepcopy(config)
    derived["output"]["base_dir"] = str(cell)
    derived.setdefault("input", {})["cot_features_base_dir"] = str(cell)
    base = build_feature_sets(derived, ["handcrafted"])["handcrafted"]
    existing = json.loads((cell / "ml_metrics.json").read_text(encoding="utf-8"))
    parameters = existing["handcrafted"]["xgboost"]["params"]
    records = {
        split: load_claim_records(source_root / f"claims_{split}.jsonl")
        for split in ("train", "val")
    }
    base_auc = auc_fit(
        base["train"], base["val"], base["columns"], parameters
    )
    rows = []
    for candidate_dir in sorted((cell / "candidates").glob("k_*")):
        model = load_model(
            candidate_dir / "cluster_model.json",
            candidate_dir / "centroids.npz",
        )
        assignments = {
            split: __import__("pandas").read_parquet(
                candidate_dir / f"claim_assignments_{split}.parquet"
            )
            for split in ("train", "val")
        }
        for encoding in ENCODINGS:
            cot = {
                split: features_from_assignments(
                    records[split], assignments[split], model, encoding=encoding
                )
                for split in ("train", "val")
            }
            ranking = rank_features_by_train_mi(cot["train"], encoding=encoding)
            sizes = sorted({min(size, len(ranking)) for size in TOP_K} | {len(ranking)})
            for size in sizes:
                names = ranking[:size]
                concat = {
                    split: merge_feature_frames(base[split], cot[split])
                    for split in ("train", "val")
                }
                rows.append({
                    "candidate": candidate_dir.name,
                    "n_clusters": len(model["feature_names"]),
                    "encoding": encoding,
                    "n_features": len(names),
                    "selected_feature_names": names,
                    "validation_cot_roc_auc": auc_fit(
                        cot["train"], cot["val"], names, parameters
                    ),
                    "validation_concat_roc_auc": auc_fit(
                        concat["train"], concat["val"],
                        [*base["columns"], *names], parameters,
                    ),
                })
                print(
                    f"{cell.parent.name} {candidate_dir.name} {encoding} "
                    f"top={size} concat_auc={rows[-1]['validation_concat_roc_auc']:.6f}",
                    flush=True,
                )
    selected = choose(rows, tie_margin)
    payload = {
        "protocol": "validation-only frozen-candidate ROC-AUC sweep",
        "cell": str(cell),
        "base_feature_set": "handcrafted",
        "base_validation_roc_auc": base_auc,
        "tie_margin": tie_margin,
        "selected": selected,
        "selected_delta_vs_handcrafted": (
            selected["validation_concat_roc_auc"] - base_auc
        ),
        "rows": rows,
        "test_accessed": False,
    }
    output = cell / "cluster_auc_retuning" / "selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-root", type=Path,
        default=Path("results/isolated/derived/reviewer-v11-isolated-r2"),
    )
    parser.add_argument("--models", nargs="+", default=["qwen", "gpt_oss"])
    parser.add_argument("--tie-margin", type=float, default=0.002)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cells = [
        args.derived_root / "datafusion_default_2023" / model / "seed_17"
        for model in args.models
    ]
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "cells": [str(cell) for cell in cells],
        "metric": "validation concat ROC-AUC",
        "test_accessed": False,
    }, indent=2), flush=True)
    if not args.execute:
        return
    for cell in cells:
        run_cell(cell, args.tie_margin)


if __name__ == "__main__":
    main()

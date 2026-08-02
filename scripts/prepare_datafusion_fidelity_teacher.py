#!/usr/bin/env python3
"""Freeze a validation-selected official DF2023 teacher for fidelity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json, files_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "teacher": "official_datafusion2023_rnn", "selection_split": "val",
        "primary_metric": "roc_auc", "stochastic_axis": "teacher_seed",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    frame = pd.read_parquet(args.predictions)
    required = {"customer_id", "label", "split", "teacher_seed", "teacher_prob_0", "teacher_prob_1"}
    if not required <= set(frame):
        raise ValueError(f"Missing teacher columns: {required-set(frame)}")
    validation = frame[frame["split"] == "val"]
    metrics = []
    for seed, cell in validation.groupby("teacher_seed", sort=True):
        if cell["customer_id"].duplicated().any():
            raise ValueError(f"Duplicate validation teacher IDs for seed {seed}")
        metrics.append({
            "seed": int(seed), "n": int(len(cell)),
            "roc_auc": float(roc_auc_score(cell["label"], cell["teacher_prob_1"])),
        })
    selected = sorted(metrics, key=lambda row: (-row["roc_auc"], row["seed"]))[0]
    paths = {split: str(args.predictions) for split in ("train", "val", "test")}
    payload = {
        **plan, "selected_teacher": "official_datafusion2023_rnn",
        "selected": {
            "paths": paths, "filters": {"teacher_seed": selected["seed"]},
            "file_hashes": files_fingerprint(paths.values()),
        },
        "selected_candidate": selected, "candidate_metrics": metrics,
        "note": "Seed is frozen using validation ROC-AUC; test labels do not select it.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

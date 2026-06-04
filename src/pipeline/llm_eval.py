"""Evaluation utilities for direct LLM predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef


def split_output_path(config: dict, key: str, split: str) -> Path:
    out_dir = Path(config["output"]["base_dir"])
    base = Path(config["output"][key])
    return out_dir / f"{base.stem}_{split}{base.suffix}"


def evaluate_llm_predictions(config: dict, split: str) -> dict:
    path = split_output_path(config, "explanations", split)
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    labels = []
    preds = []
    skipped = 0
    errors = 0
    for row in rows:
        label = int(row.get("label", -1))
        pred = row.get("predicted", None)
        if row.get("error"):
            errors += 1
        if label < 0 or pred is None:
            skipped += 1
            continue
        labels.append(label)
        preds.append(int(pred))

    result = {
        "split": split,
        "n_rows": len(rows),
        "n_scored": len(labels),
        "n_skipped": skipped,
        "n_errors": errors,
        "coverage": float(len(labels) / len(rows)) if rows else 0.0,
    }

    if labels:
        y_true = np.asarray(labels, dtype=int)
        y_pred = np.asarray(preds, dtype=int)
        result.update(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            }
        )

    out_path = Path(config["output"]["base_dir"]) / f"llm_metrics_{split}.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"Saved LLM metrics for {split} -> {out_path}")
    return result

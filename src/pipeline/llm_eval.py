"""Evaluation utilities for direct LLM predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def balanced_accuracy_interval(
    y_true,
    y_pred,
    *,
    n_bootstrap: int = 1000,
    seed: int = 17,
) -> dict:
    """Paired, class-stratified client bootstrap interval."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError("Non-empty y_true/y_pred with equal length required")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    scores = []
    for _ in range(int(n_bootstrap)):
        indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True)
            for group in groups
        ])
        scores.append(balanced_accuracy_score(y_true[indices], y_pred[indices]))
    lower, upper = np.quantile(scores, [0.025, 0.975])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "confidence": 0.95,
        "method": "paired_stratified_client_bootstrap",
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }

def split_output_path(config: dict, key: str, split: str) -> Path:
    out_dir = Path(config["output"]["base_dir"])
    base = Path(config["output"][key])
    return out_dir / f"{base.stem}_{split}{base.suffix}"


def summarize_prediction_rows(rows: list[dict], *, split: str) -> dict:
    """Score only successful parsed responses while reporting failed coverage."""
    labels = []
    preds = []
    skipped = 0
    errors = 0
    for row in rows:
        label = int(row.get("label", -1))
        pred = row.get("predicted", None)
        if row.get("error"):
            errors += 1
            skipped += 1
            continue
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
        result.update({
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "positive_f1": float(
                f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            ),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        })
        if len(np.unique(y_true)) == 2:
            # Direct label-only APIs do not expose calibrated probabilities.
            # This is therefore a label-score ROC-AUC, recorded explicitly.
            result["roc_auc"] = float(roc_auc_score(y_true, y_pred))
            result["roc_auc_input"] = "hard_predicted_label"
    return result


def evaluate_llm_predictions(config: dict, split: str) -> dict:
    path = split_output_path(config, "explanations", split)
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    result = summarize_prediction_rows(rows, split=split)
    if result["n_scored"]:
        scored = [
            (int(row["label"]), int(row["predicted"]))
            for row in rows
            if not row.get("error")
            and int(row.get("label", -1)) >= 0
            and row.get("predicted") is not None
        ]
        labels, predictions = zip(*scored)
        result["balanced_accuracy_ci"] = balanced_accuracy_interval(
            labels,
            predictions,
            n_bootstrap=int(config.get("evaluation", {}).get("bootstrap_samples", 1000)),
            seed=int(config.get("evaluation", {}).get("bootstrap_seed", 17)),
        )

    out_path = Path(config["output"]["base_dir"]) / f"llm_metrics_{split}.json"
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    temporary.replace(out_path)
    print(f"Saved LLM metrics for {split} -> {out_path}")
    return result

#!/usr/bin/env python3
"""
Recompute LLM baseline metrics from split-specific explanation JSONL files.

Why this exists:
- llm_predictions_{split}.csv usually contains only the majority prediction.
- explanations_{split}.jsonl contains one row per LLM sample and therefore can be
  used to derive a soft churn score: P(churn) = fraction of valid samples predicted as label 1.
- ROC-AUC is then computed from that soft score. With one sample per client this
  degenerates to ROC-AUC over hard labels, which is still valid but less informative.

Usage:
    PYTHONPATH=. python scripts/recompute_llm_metrics.py --results-dir results/rosbank --splits val test
    PYTHONPATH=. python scripts/recompute_llm_metrics.py --results-dir results/rosbank --splits train val test --update
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_metric(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
        return round(float(value), 4)
    except Exception as exc:
        return {"error": str(exc)}


def recompute_split(path: Path) -> tuple[dict, pd.DataFrame]:
    rows = read_jsonl(path)
    by_client: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_client[int(row["customer_id"])].append(row)

    pred_records = []
    y_true, y_pred, y_score = [], [], []

    for cid, client_rows in by_client.items():
        label = int(client_rows[0].get("label", -1))
        preds = []
        raw_answers = []
        errors = []

        for row in client_rows:
            raw_answers.append(row.get("predicted_raw"))
            if row.get("error"):
                errors.append(str(row.get("error")))
            pred = row.get("predicted")
            if pred is None:
                continue
            try:
                pred_i = int(pred)
            except Exception:
                continue
            if pred_i in (0, 1):
                preds.append(pred_i)

        n_samples = len(client_rows)
        n_valid = len(preds)
        if n_valid:
            counts = Counter(preds)
            # Deterministic tie-break: label 1 only if it is strictly more frequent.
            majority_pred = 1 if counts.get(1, 0) > counts.get(0, 0) else 0
            churn_score = counts.get(1, 0) / n_valid
        else:
            majority_pred = -1
            churn_score = np.nan

        pred_records.append({
            "customer_id": cid,
            "label": label,
            "prediction": majority_pred,
            "churn_score": churn_score,
            "n_samples": n_samples,
            "n_valid_predictions": n_valid,
            "valid_prediction_rate": n_valid / n_samples if n_samples else 0.0,
            "raw_answers": raw_answers,
            "errors": errors[:3],
        })

        if label in (0, 1) and majority_pred in (0, 1) and not np.isnan(churn_score):
            y_true.append(label)
            y_pred.append(majority_pred)
            y_score.append(churn_score)

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    metrics: dict = {
        "source_file": str(path),
        "n_rows": len(rows),
        "n_clients_total": len(by_client),
        "n_clients_evaluated": int(len(y_true_arr)),
        "coverage": round(float(len(y_true_arr) / max(len(by_client), 1)), 4),
    }

    if len(y_true_arr) == 0:
        metrics["metrics_skipped"] = "no labeled clients with valid predictions"
        return metrics, pd.DataFrame(pred_records)

    metrics.update({
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true_arr, y_pred_arr)), 4),
        "f1_macro": round(float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true_arr, y_pred_arr)), 4),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
        "roc_auc": safe_metric(roc_auc_score, y_true_arr, y_score_arr),
        "roc_auc_note": "Computed from churn_score = fraction of valid LLM samples predicted as label 1. If n_explanation_samples=1, this is ROC-AUC over hard predictions.",
    })
    return metrics, pd.DataFrame(pred_records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/rosbank")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--update", action="store_true", help="Also merge the recomputed metrics into metrics_llm.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_metrics: dict[str, dict] = {}

    for split in args.splits:
        path = results_dir / f"explanations_{split}.jsonl"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        metrics, pred_df = recompute_split(path)
        all_metrics[split] = metrics

        pred_path = results_dir / f"llm_predictions_{split}_with_scores.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"[{split}] metrics: {json.dumps(metrics, ensure_ascii=False)}")
        print(f"[{split}] predictions with scores -> {pred_path}")

    out_path = results_dir / "metrics_llm_recomputed.json"
    out_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"metrics -> {out_path}")

    if args.update:
        metrics_path = results_dir / "metrics_llm.json"
        current = {}
        if metrics_path.exists():
            current = json.loads(metrics_path.read_text(encoding="utf-8"))
        for split, metrics in all_metrics.items():
            current[split] = metrics
        metrics_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"updated -> {metrics_path}")


if __name__ == "__main__":
    main()

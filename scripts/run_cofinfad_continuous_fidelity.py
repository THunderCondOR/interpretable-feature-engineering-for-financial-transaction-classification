#!/usr/bin/env python3
"""Measure how well claim-cluster features reproduce COFINFAD churn scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.common import load_json
from src.experiments.artifacts import atomic_write_json, file_sha256, fingerprint


def load_features(path: Path, expected_ids: set[str]) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(path)
    frame["customer_id"] = frame["customer_id"].astype(str)
    if frame["customer_id"].duplicated().any() or set(frame["customer_id"]) != expected_ids:
        raise ValueError(f"Feature coverage mismatch: {path}")
    feature_columns = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"customer_id", "label", "churn_probability"}
    ]
    if not feature_columns:
        raise ValueError(f"No numeric claim features in {path}")
    return frame[["customer_id", *feature_columns]].copy(), feature_columns


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "spearman_r": float(spearmanr(y_true, y_pred).statistic),
        "within_0_05": float((np.abs(y_true - y_pred) <= 0.05).mean()),
        "within_0_10": float((np.abs(y_true - y_pred) <= 0.10).mean()),
    }


def confidence_curve(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    confidence = np.maximum(y_true, 1.0 - y_true)
    rows = []
    for threshold in np.arange(0.50, 0.951, 0.05):
        selected = confidence >= threshold
        rows.append({
            "teacher_confidence_threshold": float(round(threshold, 2)),
            "coverage": float(selected.mean()),
            "n_clients": int(selected.sum()),
            "mae": float(mean_absolute_error(y_true[selected], y_pred[selected])) if selected.any() else None,
            "r2": float(r2_score(y_true[selected], y_pred[selected])) if selected.sum() >= 2 else None,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-features", required=True, type=Path)
    parser.add_argument("--val-features", required=True, type=Path)
    parser.add_argument("--test-features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    plan = {
        "mode": "execute" if args.execute else "dry-run", "dataset": manifest.get("dataset"),
        "protocol": manifest.get("protocol"), "teacher": "published_churn_probability",
        "selection_split": "val", "report_split": "test",
        "surrogates": ["ridge", "extra_trees"],
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if manifest.get("dataset") != "cofinfad_operational_fidelity":
        raise ValueError("Wrong benchmark manifest")
    scores = pd.read_csv(manifest["teacher_scores"], dtype={"customer_id": str})
    cells = {}
    for split, path in (("train", args.train_features), ("val", args.val_features), ("test", args.test_features)):
        identifiers = set(json.loads(Path(manifest["roles"][split]).read_text()))
        features, columns = load_features(path, identifiers)
        merged = features.merge(
            scores[["customer_id", "churn_probability"]], on="customer_id", validate="one_to_one"
        )
        cells[split] = (merged, columns)
    if not (cells["train"][1] == cells["val"][1] == cells["test"][1]):
        raise ValueError("Claim feature schemas differ across splits")
    columns = cells["train"][1]
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=5, max_features=0.7,
            random_state=args.seed, n_jobs=-1,
        ),
    }
    validation = []
    for name, model in candidates.items():
        model.fit(cells["train"][0][columns], cells["train"][0]["churn_probability"])
        row = {"candidate": name, **metrics(
            cells["val"][0]["churn_probability"].to_numpy(),
            model.predict(cells["val"][0][columns]),
        )}
        validation.append(row)
    selected_name = sorted(validation, key=lambda row: (-row["r2"], row["mae"], row["candidate"]))[0]["candidate"]
    train_val = pd.concat([cells["train"][0], cells["val"][0]], ignore_index=True)
    selected = candidates[selected_name]
    selected.fit(train_val[columns], train_val["churn_probability"])
    test = cells["test"][0].copy()
    test["surrogate_churn_probability"] = np.clip(selected.predict(test[columns]), 0, 1)
    test_metrics = metrics(test["churn_probability"].to_numpy(), test["surrogate_churn_probability"].to_numpy())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "continuous_fidelity_test_predictions.csv"
    test[["customer_id", "churn_probability", "surrogate_churn_probability"]].to_csv(predictions, index=False)
    payload = {
        **plan, "status": "completed", "feature_columns": columns,
        "feature_files_sha256": {split: file_sha256(path) for split, path in (
            ("train", args.train_features), ("val", args.val_features), ("test", args.test_features)
        )},
        "validation_candidates": validation, "selected_surrogate": selected_name,
        "test_metrics": test_metrics,
        "test_teacher_confidence_curve": confidence_curve(
            test["churn_probability"].to_numpy(), test["surrogate_churn_probability"].to_numpy()
        ),
        "interpretation": "Predictive/surrogate fidelity to a published derived score, not causal faithfulness.",
        "predictions_sha256": file_sha256(predictions),
    }
    payload["result_signature"] = fingerprint(payload)
    atomic_write_json(args.output_dir / "continuous_fidelity_metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

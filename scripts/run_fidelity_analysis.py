"""Train claim-based surrogates and measure fidelity to a frozen teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from src.evaluation.reviewer_metrics import surrogate_fidelity


def load_cell(features: Path, teacher: Path):
    frame = pd.read_parquet(features)
    scores = pd.read_csv(teacher)
    return frame.merge(scores, on=["customer_id", "label"], validate="one_to_one")


def teacher_probabilities(frame):
    columns = sorted(column for column in frame if column.startswith("teacher_prob_"))
    if not columns:
        raise ValueError("Teacher CSV must contain teacher_prob_<class> columns")
    return frame[columns].to_numpy(float)


def models(seed, n_classes):
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "xgboost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, random_state=seed, tree_method="hist",
            objective=objective, eval_metric="logloss", n_jobs=2,
        ),
        "shallow_tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=seed),
    }


def aligned_probabilities(model, values, n_classes):
    raw = model.predict_proba(values)
    aligned = np.zeros((len(values), n_classes), dtype=float)
    aligned[:, model.classes_.astype(int)] = raw
    return aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True, type=Path)
    parser.add_argument("--val-features", required=True, type=Path)
    parser.add_argument("--test-features", required=True, type=Path)
    parser.add_argument("--train-teacher", required=True, type=Path)
    parser.add_argument("--val-teacher", required=True, type=Path)
    parser.add_argument("--test-teacher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = {
        "teacher_selection": "must be frozen using validation before this script",
        "surrogates": ["logistic_regression", "xgboost", "shallow_tree"],
        "metrics": ["hard agreement", "probability MAE/RMSE", "Jensen-Shannon", "outcome quadrants"],
    }
    if not args.execute:
        print(json.dumps({"mode": "dry-run", **plan}, indent=2))
        return

    train = load_cell(args.train_features, args.train_teacher)
    val = load_cell(args.val_features, args.val_teacher)
    test = load_cell(args.test_features, args.test_teacher)
    features = [column for column in train if column.startswith("cot_")]
    n_classes = teacher_probabilities(train).shape[1]
    hard_teacher = teacher_probabilities(train).argmax(1)
    results, predictions = {}, []
    fitted = models(args.seed, n_classes)
    for name, model in fitted.items():
        model.fit(train[features], hard_teacher)
        results[name] = {}
        for split, frame in (("val", val), ("test", test)):
            probabilities = aligned_probabilities(model, frame[features], n_classes)
            results[name][split] = surrogate_fidelity(
                teacher_probabilities(frame), probabilities, frame["label"].to_numpy(int)
            )
            for row_index, row in frame.reset_index(drop=True).iterrows():
                predictions.append({
                    "model": name, "split": split, "customer_id": int(row.customer_id),
                    "label": int(row.label),
                    **{f"surrogate_prob_{i}": float(probabilities[row_index, i]) for i in range(n_classes)},
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fidelity_metrics.json").write_text(json.dumps({"protocol": plan, "results": results}, indent=2), encoding="utf-8")
    pd.DataFrame(predictions).to_csv(args.output_dir / "surrogate_predictions.csv", index=False)
    (args.output_dir / "shallow_tree.txt").write_text(export_text(fitted["shallow_tree"], feature_names=features), encoding="utf-8")
    print(f"Saved fidelity analysis -> {args.output_dir}")


if __name__ == "__main__":
    main()

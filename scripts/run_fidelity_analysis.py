"""Train claim-based surrogates and measure fidelity to a frozen teacher."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.reviewer_metrics import cluster_occlusion, surrogate_fidelity
from src.experiments.artifacts import atomic_write_json, files_fingerprint


def read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    return pd.read_csv(path)


def load_cell(
    features: Path,
    teacher: Path,
    *,
    filters: dict | None = None,
    split: str | None = None,
):
    frame = pd.read_parquet(features)
    scores = read_predictions(teacher)
    for column, value in (filters or {}).items():
        if column not in scores:
            raise ValueError(f"Missing teacher filter column: {column}")
        scores = scores[scores[column] == value]
    if split and "split" in scores:
        scores = scores[scores["split"] == split]
    if frame.empty or scores.empty:
        raise ValueError(f"Empty feature/teacher cell for split={split}")
    keys = ["customer_id", "label"]
    if frame[keys].duplicated().any():
        raise ValueError(f"Duplicate feature customer/label rows for split={split}")
    if scores[keys].duplicated().any():
        raise ValueError(f"Duplicate teacher customer/label rows for split={split}")
    feature_keys = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
    teacher_keys = set(map(tuple, scores[keys].itertuples(index=False, name=None)))
    if feature_keys != teacher_keys:
        missing = len(feature_keys - teacher_keys)
        extra = len(teacher_keys - feature_keys)
        raise ValueError(
            f"Teacher coverage mismatch for split={split}: missing={missing}, extra={extra}"
        )
    keep = ["customer_id", "label", *_probability_columns(scores)]
    merged = frame.merge(scores[keep], on=keys, validate="one_to_one")
    if len(merged) != len(frame):
        raise ValueError(f"Teacher merge lost rows for split={split}")
    return merged


def validate_selected_teacher(selection: dict, paths: dict[str, str]) -> None:
    expected = selection.get("selected", {}).get("file_hashes")
    if not expected:
        raise ValueError("Teacher selection is missing selected.file_hashes")
    actual = files_fingerprint(paths.values())
    if actual != expected:
        raise ValueError("Selected teacher files changed after validation selection")


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    for prefix in ("teacher_prob_", "probability_"):
        columns = [column for column in frame if column.startswith(prefix)]
        if columns:
            return sorted(columns, key=lambda column: int(column[len(prefix):]))
    raise ValueError(
        "Teacher predictions require teacher_prob_<class> or probability_<class> columns"
    )


def teacher_probabilities(frame):
    values = frame[_probability_columns(frame)].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-4)
    ):
        raise ValueError("Invalid teacher probability distributions")
    return values


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


def fit_soft_classifier(model, values, probabilities):
    """Fit a classifier to teacher probabilities via weighted label expansion."""
    probabilities = np.asarray(probabilities, dtype=float)
    n_classes = probabilities.shape[1]
    expanded_values = np.repeat(np.asarray(values), n_classes, axis=0)
    expanded_labels = np.tile(np.arange(n_classes), len(probabilities))
    weights = probabilities.reshape(-1)
    keep = weights > 1e-12
    model.fit(
        expanded_values[keep],
        expanded_labels[keep],
        sample_weight=weights[keep],
    )
    return model


def aligned_probabilities(model, values, n_classes):
    raw = model.predict_proba(values)
    aligned = np.zeros((len(values), n_classes), dtype=float)
    aligned[:, model.classes_.astype(int)] = raw
    return aligned



def tree_decision_paths(model, frame, features, display_names):
    values = frame[features].to_numpy(dtype=float)
    indicator = model.decision_path(values)
    leaves = model.apply(values)
    tree = model.tree_
    records = []
    for row_index, row in frame.reset_index(drop=True).iterrows():
        steps = []
        node_ids = indicator.indices[
            indicator.indptr[row_index]:indicator.indptr[row_index + 1]
        ]
        for node_id in node_ids:
            feature_index = int(tree.feature[node_id])
            if feature_index < 0:
                continue
            threshold = float(tree.threshold[node_id])
            value = float(values[row_index, feature_index])
            steps.append({
                "node_id": int(node_id),
                "feature": features[feature_index],
                "semantic_name": display_names[feature_index],
                "value": value,
                "threshold": threshold,
                "operator": "<=" if value <= threshold else ">",
            })
        leaf_id = int(leaves[row_index])
        class_weights = tree.value[leaf_id][0].astype(float)
        probabilities = class_weights / max(float(class_weights.sum()), 1e-12)
        records.append({
            "customer_id": int(row.customer_id),
            "label": int(row.label),
            "leaf_id": leaf_id,
            "predicted_class": int(model.classes_[probabilities.argmax()]),
            "leaf_class_weights": class_weights.tolist(),
            "leaf_probabilities": probabilities.tolist(),
            "steps": steps,
        })
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True, type=Path)
    parser.add_argument("--val-features", required=True, type=Path)
    parser.add_argument("--test-features", required=True, type=Path)
    parser.add_argument("--train-teacher", type=Path)
    parser.add_argument("--val-teacher", type=Path)
    parser.add_argument("--test-teacher", type=Path)
    parser.add_argument("--teacher-selection", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cluster-metadata", type=Path)
    parser.add_argument("--max-shown-clusters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    teacher_filters = {}
    if args.teacher_selection:
        selection = json.loads(args.teacher_selection.read_text(encoding="utf-8"))
        paths = selection["selected"]["paths"]
        validate_selected_teacher(selection, paths)
        teacher_filters = selection["selected"].get("filters", {})
        args.train_teacher = Path(paths["train"])
        args.val_teacher = Path(paths["val"])
        args.test_teacher = Path(paths["test"])
    if not all((args.train_teacher, args.val_teacher, args.test_teacher)):
        parser.error(
            "provide --teacher-selection or all of "
            "--train-teacher/--val-teacher/--test-teacher"
        )

    plan = {
        "teacher_selection": "frozen validation-selected teacher inputs",
        "training_target": "teacher probability distribution via weighted label expansion",
        "surrogates": ["logistic_regression", "xgboost", "shallow_tree"],
        "metrics": ["hard agreement", "probability MAE/RMSE", "Jensen-Shannon", "outcome quadrants"],
    }
    if not args.execute:
        print(json.dumps({"mode": "dry-run", **plan}, indent=2))
        return

    train = load_cell(args.train_features, args.train_teacher, filters=teacher_filters, split="train")
    val = load_cell(args.val_features, args.val_teacher, filters=teacher_filters, split="val")
    test = load_cell(args.test_features, args.test_teacher, filters=teacher_filters, split="test")
    features = [column for column in train if column.startswith("cot_")]
    n_classes = teacher_probabilities(train).shape[1]
    train_teacher_probabilities = teacher_probabilities(train)
    results, predictions, occlusions = {}, [], []
    fitted = models(args.seed, n_classes)
    for name, model in fitted.items():
        fit_soft_classifier(model, train[features], train_teacher_probabilities)
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
                if split == "test":
                    values = frame.iloc[row_index][features].to_numpy(dtype=float)
                    active = np.flatnonzero(values != 0)[: args.max_shown_clusters].tolist()
                    if active:
                        occlusions.append({
                            "model": name,
                            "customer_id": int(row.customer_id),
                            "shown_features": [features[index] for index in active],
                            **cluster_occlusion(
                                lambda x: aligned_probabilities(model, x, n_classes),
                                values,
                                active,
                            ),
                        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output_dir / "fidelity_metrics.json",
        {"protocol": plan, "results": results},
    )
    predictions_path = args.output_dir / "surrogate_predictions.csv"
    predictions_tmp = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
    pd.DataFrame(predictions).to_csv(predictions_tmp, index=False)
    predictions_tmp.replace(predictions_path)
    occlusion_path = args.output_dir / "cluster_occlusion.jsonl"
    occlusion_tmp = occlusion_path.with_suffix(occlusion_path.suffix + ".tmp")
    with open(occlusion_tmp, "w", encoding="utf-8") as file:
        for record in occlusions:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    occlusion_tmp.replace(occlusion_path)
    tree_names = features
    if args.cluster_metadata:
        payload = json.loads(args.cluster_metadata.read_text(encoding="utf-8"))
        metadata = payload.get("cluster_meta", payload) if isinstance(payload, dict) else payload
        medoids = {
            str(row.get("feature")): str(row.get("medoid") or row.get("feature"))
            for row in metadata
        }
        tree_names = [medoids.get(feature, feature)[:120] for feature in features]
    tree_path = args.output_dir / "shallow_tree.txt"
    tree_tmp = tree_path.with_suffix(tree_path.suffix + ".tmp")
    tree_tmp.write_text(
        export_text(fitted["shallow_tree"], feature_names=tree_names),
        encoding="utf-8",
    )
    tree_tmp.replace(tree_path)
    decision_path = args.output_dir / "tree_decision_paths.jsonl"
    decision_tmp = decision_path.with_suffix(decision_path.suffix + ".tmp")
    with open(decision_tmp, "w", encoding="utf-8") as file:
        for record in tree_decision_paths(
            fitted["shallow_tree"], test, features, tree_names
        ):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    decision_tmp.replace(decision_path)
    print(f"Saved fidelity analysis -> {args.output_dir}")


if __name__ == "__main__":
    main()

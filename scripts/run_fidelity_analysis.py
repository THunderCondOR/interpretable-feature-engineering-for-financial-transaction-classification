"""Train claim-based surrogates and measure fidelity to a frozen teacher."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.reviewer_metrics import cluster_occlusion, surrogate_fidelity
from src.data.entity_ids import canonical_entity_series
from src.experiments.artifacts import (
    atomic_write_json,
    files_fingerprint,
)
from src.experiments.derived_artifacts import (
    compatible_stage,
    complete_stage,
    stage_identity,
)


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
    # Parquet feature artifacts preserve numeric IDs, whereas imported teacher
    # predictions may carry the same IDs as strings.  Compare and merge using
    # the shared canonical representation while preserving opaque IDs.
    frame = frame.copy()
    scores = scores.copy()
    frame["customer_id"] = canonical_entity_series(frame["customer_id"])
    scores["customer_id"] = canonical_entity_series(scores["customer_id"])
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


def select_binary_teacher_threshold(probabilities, labels) -> dict[str, float]:
    """Freeze a non-degenerate teacher decision threshold on validation."""
    positive = np.asarray(probabilities, dtype=float)[:, 1]
    labels = np.asarray(labels, dtype=int)
    candidates = np.unique(
        np.r_[0.0, 0.5, 1.0, np.quantile(positive, np.linspace(0.01, 0.99, 99))]
    )
    rows = []
    for threshold in candidates:
        prediction = (positive >= float(threshold)).astype(int)
        rows.append({
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            "positive_rate": float(prediction.mean()),
        })
    return max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            -abs(row["positive_rate"] - float(labels.mean())),
            -abs(row["threshold"] - 0.5),
        ),
    )


def select_binary_surrogate_threshold(
    teacher_probabilities_value, surrogate_probabilities_value, teacher_threshold
) -> dict[str, float]:
    teacher_prediction = (
        np.asarray(teacher_probabilities_value, dtype=float)[:, 1]
        >= float(teacher_threshold)
    ).astype(int)
    positive = np.asarray(surrogate_probabilities_value, dtype=float)[:, 1]
    candidates = np.unique(
        np.r_[0.0, 0.5, 1.0, np.quantile(positive, np.linspace(0.01, 0.99, 99))]
    )
    rows = []
    for threshold in candidates:
        prediction = (positive >= float(threshold)).astype(int)
        rows.append({
            "threshold": float(threshold),
            "hard_agreement": float((prediction == teacher_prediction).mean()),
        })
    return max(
        rows,
        key=lambda row: (row["hard_agreement"], -abs(row["threshold"] - 0.5)),
    )


def thresholded_binary_fidelity(
    teacher, surrogate, labels, *, teacher_threshold, surrogate_threshold
) -> dict[str, float]:
    teacher_prediction = (
        np.asarray(teacher, dtype=float)[:, 1] >= float(teacher_threshold)
    ).astype(int)
    surrogate_prediction = (
        np.asarray(surrogate, dtype=float)[:, 1] >= float(surrogate_threshold)
    ).astype(int)
    labels = np.asarray(labels, dtype=int)
    agreement = teacher_prediction == surrogate_prediction
    teacher_correct = teacher_prediction == labels
    surrogate_correct = surrogate_prediction == labels
    return {
        "teacher_threshold": float(teacher_threshold),
        "surrogate_threshold": float(surrogate_threshold),
        "hard_agreement": float(agreement.mean()),
        "teacher_positive_rate": float(teacher_prediction.mean()),
        "surrogate_positive_rate": float(surrogate_prediction.mean()),
        "agree_and_correct": float((agreement & teacher_correct).mean()),
        "agree_and_wrong": float((agreement & ~teacher_correct).mean()),
        "teacher_only_correct": float((teacher_correct & ~surrogate_correct).mean()),
        "surrogate_only_correct": float((~teacher_correct & surrogate_correct).mean()),
    }


def surrogate_specs(seed: int, n_classes: int) -> list[dict[str, Any]]:
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    specs: list[dict[str, Any]] = []
    for complexity, regularization in enumerate((0.1, 1.0)):
        specs.append({
            "family": "logistic_regression",
            "name": f"logistic_c_{regularization:g}",
            "complexity": complexity,
            "factory": lambda c=regularization: LogisticRegression(
                C=c, max_iter=2000, random_state=seed
            ),
        })
    for complexity, depth in enumerate((2, 4)):
        specs.append({
            "family": "xgboost",
            "name": f"xgboost_depth_{depth}",
            "complexity": complexity,
            "factory": lambda d=depth: XGBClassifier(
                n_estimators=250, max_depth=d, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                random_state=seed, tree_method="hist",
                objective=objective, eval_metric="logloss", n_jobs=2,
            ),
        })
    for complexity, (depth, leaf) in enumerate(((3, 40), (4, 20))):
        specs.append({
            "family": "shallow_tree",
            "name": f"tree_depth_{depth}_leaf_{leaf}",
            "complexity": complexity,
            "factory": lambda d=depth, l=leaf: DecisionTreeClassifier(
                max_depth=d, min_samples_leaf=l, random_state=seed
            ),
        })
    return specs


def rank_features_for_teacher(
    values: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Rank train features by association with teacher probabilities only."""
    x = np.asarray(values, dtype=float)
    targets = np.asarray(probabilities, dtype=float)
    centered_x = x - x.mean(axis=0, keepdims=True)
    x_norm = np.sqrt(np.square(centered_x).sum(axis=0))
    scores = np.zeros(x.shape[1], dtype=float)
    for class_index in range(targets.shape[1]):
        target = targets[:, class_index]
        centered_target = target - target.mean()
        denominator = x_norm * np.sqrt(np.square(centered_target).sum())
        correlation = np.divide(
            np.abs(centered_x.T @ centered_target),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        scores = np.maximum(scores, correlation)
    return np.argsort(-scores, kind="stable")


def select_candidate(
    candidates: list[dict[str, Any]],
    *,
    tie_margin: float,
) -> dict[str, Any]:
    selection_key = (
        "thresholded_hard_agreement"
        if all("thresholded_hard_agreement" in row for row in candidates)
        else "hard_agreement"
    )
    best = max(row[selection_key] for row in candidates)
    eligible = [
        row for row in candidates
        if best - row[selection_key] <= float(tie_margin)
    ]
    return sorted(
        eligible,
        key=lambda row: (
            row["jensen_shannon_divergence"],
            row["probability_mae"],
            row["n_features"],
            row["complexity"],
            row["candidate"],
        ),
    )[0]


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


def bootstrap_fidelity(
    teacher, surrogate, labels, *, samples=1000, seed=17
):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        metrics = surrogate_fidelity(
            teacher[indices], surrogate[indices], labels[indices]
        )
        rows.append(metrics)
    result = {}
    for metric in (
        "hard_agreement",
        "probability_mae",
        "probability_rmse",
        "jensen_shannon_divergence",
    ):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
    return result


def ranked_occlusion(
    model,
    values,
    features,
    n_classes,
    max_clusters,
    *,
    encoding="binary",
):
    active = np.flatnonzero(values != 0)
    if not len(active):
        return None
    row = np.asarray(values, dtype=float).reshape(1, -1)
    base = aligned_probabilities(model, row, n_classes)[0]
    predicted = int(base.argmax())
    masked = np.repeat(row, len(active), axis=0)
    masked[np.arange(len(active)), active] = 0
    if encoding == "normalized_count":
        totals = masked.sum(axis=1, keepdims=True)
        masked = np.divide(
            masked,
            totals,
            out=np.zeros_like(masked),
            where=totals > 0,
        )
    impacts = base[predicted] - aligned_probabilities(
        model, masked, n_classes
    )[:, predicted]
    order = np.argsort(-np.abs(impacts), kind="stable")
    ranked = active[order][:max_clusters].tolist()
    analyses = {}
    for count in (1, 3, 5, 10):
        shown = ranked[:count]
        if shown:
            analyses[str(count)] = cluster_occlusion(
                lambda x: aligned_probabilities(model, x, n_classes),
                values,
                shown,
                encoding=encoding,
            )
    return {
        "ranked_features": [features[index] for index in ranked],
        "ranked_signed_impacts": [
            float(impacts[index]) for index in order[:max_clusters]
        ],
        "top_k": analyses,
    }



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
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutation-controls", type=int, default=20)
    parser.add_argument("--teacher-seed", type=int, default=17)
    parser.add_argument("--surrogate-seed", type=int, default=17)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Deprecated alias for --surrogate-seed.",
    )
    parser.add_argument(
        "--feature-counts", nargs="+", type=int, default=[50, 100],
    )
    parser.add_argument("--selection-tie-margin", type=float, default=0.005)
    parser.add_argument(
        "--encoding",
        choices=("binary", "raw_count", "normalized_count"),
        default="binary",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.seed is not None:
        args.surrogate_seed = int(args.seed)
    teacher_filters = {}
    if args.teacher_selection:
        selection = json.loads(args.teacher_selection.read_text(encoding="utf-8"))
        paths = selection["selected"]["paths"]
        validate_selected_teacher(selection, paths)
        teacher_filters = selection["selected"].get("filters", {})
        if (
            "seed" in teacher_filters
            and int(teacher_filters["seed"]) != int(args.teacher_seed)
        ):
            raise ValueError(
                "Teacher selection filter seed does not match --teacher-seed"
            )
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
        "surrogate_selection": (
            "validation hard agreement; Jensen-Shannon and simplicity tie-breaks"
        ),
        "surrogates": ["logistic_regression", "xgboost", "shallow_tree"],
        "feature_selection": "train-only association with teacher probabilities",
        "teacher_seed": int(args.teacher_seed),
        "surrogate_seed": int(args.surrogate_seed),
        "encoding": args.encoding,
        "metrics": ["hard agreement", "probability MAE/RMSE", "Jensen-Shannon", "outcome quadrants"],
    }
    if not args.execute:
        print(json.dumps({"mode": "dry-run", **plan}, indent=2))
        return

    input_paths = [
        args.train_features, args.val_features, args.test_features,
        args.train_teacher, args.val_teacher, args.test_teacher,
        Path(__file__),
        REPO_ROOT / "src/evaluation/reviewer_metrics.py",
    ]
    if args.teacher_selection:
        input_paths.append(args.teacher_selection)
    if args.cluster_metadata:
        input_paths.append(args.cluster_metadata)
    identity = stage_identity(
        stage="fidelity_v2",
        source={
            "teacher_selection": (
                str(args.teacher_selection) if args.teacher_selection else None
            ),
            "teacher_seed": int(args.teacher_seed),
        },
        inputs=files_fingerprint(input_paths),
        configuration={
            **plan,
            "feature_counts": args.feature_counts,
            "selection_tie_margin": args.selection_tie_margin,
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_controls": args.permutation_controls,
        },
        repo_root=REPO_ROOT,
    )
    stage_path = args.output_dir / "fidelity_stage.json"
    if compatible_stage(stage_path, identity):
        print(f"Reused compatible fidelity analysis -> {args.output_dir}")
        return

    train = load_cell(args.train_features, args.train_teacher, filters=teacher_filters, split="train")
    val = load_cell(args.val_features, args.val_teacher, filters=teacher_filters, split="val")
    test = load_cell(args.test_features, args.test_teacher, filters=teacher_filters, split="test")
    features = [column for column in train if column.startswith("cot_")]
    if not features:
        raise ValueError("Fidelity requires at least one cot_ feature")
    for split, frame in (("train", train), ("val", val), ("test", test)):
        if list(frame[features].columns) != features:
            raise ValueError(f"Feature order mismatch for split={split}")
        if not np.isfinite(frame[features].to_numpy(float)).all():
            raise ValueError(f"Non-finite claim features for split={split}")
    n_classes = teacher_probabilities(train).shape[1]
    train_teacher_probabilities = teacher_probabilities(train)
    train_values = train[features].to_numpy(float)
    val_values = val[features].to_numpy(float)
    teacher_threshold_selection = None
    if n_classes == 2:
        teacher_threshold_selection = select_binary_teacher_threshold(
            teacher_probabilities(val), val["label"].to_numpy(int)
        )
    ranking = rank_features_for_teacher(
        train_values, train_teacher_probabilities
    )
    feature_counts = sorted({
        min(int(value), len(features))
        for value in [*args.feature_counts, len(features)]
        if int(value) > 0
    })
    candidate_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in surrogate_specs(args.surrogate_seed, n_classes):
        for count in feature_counts:
            indices = ranking[:count]
            model = fit_soft_classifier(
                spec["factory"](),
                train_values[:, indices],
                train_teacher_probabilities,
            )
            probabilities = aligned_probabilities(
                model, val_values[:, indices], n_classes
            )
            metrics = surrogate_fidelity(
                teacher_probabilities(val),
                probabilities,
                val["label"].to_numpy(int),
            )
            surrogate_threshold_selection = None
            thresholded = None
            if teacher_threshold_selection is not None:
                surrogate_threshold_selection = select_binary_surrogate_threshold(
                    teacher_probabilities(val), probabilities,
                    teacher_threshold_selection["threshold"],
                )
                thresholded = thresholded_binary_fidelity(
                    teacher_probabilities(val), probabilities,
                    val["label"].to_numpy(int),
                    teacher_threshold=teacher_threshold_selection["threshold"],
                    surrogate_threshold=surrogate_threshold_selection["threshold"],
                )
            candidate_rows.setdefault(spec["family"], []).append({
                "candidate": spec["name"],
                "family": spec["family"],
                "complexity": int(spec["complexity"]),
                "n_features": int(count),
                **(
                    {"thresholded_hard_agreement": thresholded["hard_agreement"]}
                    if thresholded is not None else {}
                ),
                **{
                    key: metrics[key] for key in (
                        "hard_agreement",
                        "probability_mae",
                        "probability_rmse",
                        "jensen_shannon_divergence",
                    )
                },
                "_model": model,
                "_indices": indices,
                "_spec": spec,
                "_surrogate_threshold": (
                    surrogate_threshold_selection["threshold"]
                    if surrogate_threshold_selection is not None else None
                ),
            })
    selected_internal = {
        family: select_candidate(
            rows, tie_margin=args.selection_tie_margin
        )
        for family, rows in candidate_rows.items()
    }
    overall = select_candidate(
        list(selected_internal.values()),
        tie_margin=args.selection_tie_margin,
    )
    selected_surrogate = str(overall["family"])
    clean = lambda row: {
        key: value for key, value in row.items() if not key.startswith("_")
    }
    selection_payload = {
        "selection_split": "validation",
        "primary_metric": (
            "validation-thresholded hard agreement"
            if teacher_threshold_selection is not None else "hard_agreement"
        ),
        "teacher_threshold_selection": teacher_threshold_selection,
        "tie_margin": args.selection_tie_margin,
        "selected_surrogate": selected_surrogate,
        "selected_candidate": clean(overall),
        "selected_by_family": {
            family: clean(row)
            for family, row in selected_internal.items()
        },
        "candidates": {
            family: [clean(row) for row in rows]
            for family, rows in candidate_rows.items()
        },
    }

    results, predictions, occlusions = {}, [], []
    for family, selected in selected_internal.items():
        model = selected["_model"]
        indices = selected["_indices"]
        selected_features = [features[index] for index in indices]
        results[family] = {
            "selected_candidate": clean(selected),
        }
        for split, frame in (("val", val), ("test", test)):
            values = frame[features].to_numpy(float)[:, indices]
            probabilities = aligned_probabilities(model, values, n_classes)
            results[family][split] = surrogate_fidelity(
                teacher_probabilities(frame), probabilities,
                frame["label"].to_numpy(int),
            )
            if teacher_threshold_selection is not None:
                results[family][split]["thresholded_decision_fidelity"] = (
                    thresholded_binary_fidelity(
                        teacher_probabilities(frame), probabilities,
                        frame["label"].to_numpy(int),
                        teacher_threshold=teacher_threshold_selection["threshold"],
                        surrogate_threshold=selected["_surrogate_threshold"],
                    )
                )
            results[family][split]["bootstrap"] = bootstrap_fidelity(
                teacher_probabilities(frame),
                probabilities,
                frame["label"].to_numpy(int),
                samples=args.bootstrap_samples,
                seed=args.surrogate_seed,
            )
            for row_index, row in frame.reset_index(drop=True).iterrows():
                predictions.append({
                    "model": family,
                    "candidate": selected["candidate"],
                    "split": split,
                    "customer_id": int(row.customer_id),
                    "label": int(row.label),
                    **{
                        f"surrogate_prob_{i}": float(
                            probabilities[row_index, i]
                        )
                        for i in range(n_classes)
                    },
                })
                if split == "test":
                    analysis = ranked_occlusion(
                        model, values[row_index], selected_features,
                        n_classes, args.max_shown_clusters,
                        encoding=args.encoding,
                    )
                    if analysis:
                        occlusions.append({
                            "model": family,
                            "candidate": selected["candidate"],
                            "customer_id": int(row.customer_id),
                            **analysis,
                        })
    prior = train_teacher_probabilities.mean(axis=0)
    prior_test = np.repeat(prior[None, :], len(test), axis=0)
    results["prior_control"] = surrogate_fidelity(
        teacher_probabilities(test), prior_test,
        test["label"].to_numpy(int),
    )
    rng = np.random.default_rng(args.surrogate_seed)
    permutation_results = []
    control_spec = overall["_spec"]
    control_indices = overall["_indices"]
    for permutation in range(args.permutation_controls):
        shuffled = train_teacher_probabilities[
            rng.permutation(len(train_teacher_probabilities))
        ]
        model = fit_soft_classifier(
            control_spec["factory"](),
            train_values[:, control_indices],
            shuffled,
        )
        probabilities = aligned_probabilities(
            model,
            test[features].to_numpy(float)[:, control_indices],
            n_classes,
        )
        permutation_results.append(surrogate_fidelity(
            teacher_probabilities(test), probabilities,
            test["label"].to_numpy(int),
        ))
    results["permutation_control"] = {
        "matched_surrogate": selected_surrogate,
        "matched_candidate": overall["candidate"],
        "n": args.permutation_controls,
        "hard_agreement": [
            row["hard_agreement"] for row in permutation_results
        ],
        "probability_mae": [
            row["probability_mae"] for row in permutation_results
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output_dir / "fidelity_metrics.json",
        {
            "protocol": plan,
            "surrogate_selection": selection_payload,
            "results": results,
        },
    )
    atomic_write_json(
        args.output_dir / "surrogate_selection.json",
        selection_payload,
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
    tree_selected = selected_internal["shallow_tree"]
    tree_indices = tree_selected["_indices"]
    tree_features = [features[index] for index in tree_indices]
    tree_names = tree_features
    if args.cluster_metadata:
        payload = json.loads(args.cluster_metadata.read_text(encoding="utf-8"))
        metadata = payload.get("cluster_meta", payload) if isinstance(payload, dict) else payload
        medoids = {
            str(row.get("feature")): str(row.get("medoid") or row.get("feature"))
            for row in metadata
        }
        tree_names = [
            medoids.get(feature, feature)[:120] for feature in tree_features
        ]
    tree_path = args.output_dir / "shallow_tree.txt"
    tree_tmp = tree_path.with_suffix(tree_path.suffix + ".tmp")
    tree_tmp.write_text(
        export_text(
            tree_selected["_model"], feature_names=tree_names
        ),
        encoding="utf-8",
    )
    tree_tmp.replace(tree_path)
    decision_path = args.output_dir / "tree_decision_paths.jsonl"
    decision_tmp = decision_path.with_suffix(decision_path.suffix + ".tmp")
    with open(decision_tmp, "w", encoding="utf-8") as file:
        for record in tree_decision_paths(
            tree_selected["_model"],
            test[["customer_id", "label", *tree_features]],
            tree_features,
            tree_names,
        ):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    decision_tmp.replace(decision_path)
    complete_stage(
        stage_path,
        identity,
        outputs=[
            args.output_dir / "fidelity_metrics.json",
            args.output_dir / "surrogate_selection.json",
            predictions_path,
            occlusion_path,
            tree_path,
            decision_path,
        ],
        metrics={
            "selected_surrogate": selected_surrogate,
            "selected_candidate": clean(overall),
        },
    )
    print(f"Saved fidelity analysis -> {args.output_dir}")


if __name__ == "__main__":
    main()

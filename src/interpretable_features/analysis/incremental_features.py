"""Validation-only retuning of incremental claim-cluster value.

The runner never rebuilds explanations, claims, embeddings, or clusters.  It
uses frozen feature matrices, ranks cluster columns by their association with
out-of-fold residuals of a non-claim baseline, keeps K=0 as an explicit
control, and selects the classifier/K/blend on validation only.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from interpretable_features.evaluation.cluster_cleanup import (
    CLEANUP_VARIANTS,
    select_clean_cluster_features,
)
from interpretable_features.experiments.artifacts import atomic_write_json, fingerprint
from interpretable_features.models.ml_baseline import (
    build_feature_sets,
    merge_feature_frames,
    split_xy,
)

BASE_FEATURES = ("standard", "handcrafted", "standard_profile", "all_nonclaim")
DEFAULT_K = (0, 5, 10, 20, 50, 100, 200)
SEEDS = (17, 101, 947)
PROTOCOL_VERSION = 1
DATASET_PRIMARY_METRICS = {
    "rosbank": "roc_auc",
    "gender": "balanced_accuracy",
    "age": "f1_macro",
    "berka": "positive_f1",
    "datafusion_default_2023": "roc_auc",
}


def metric_name(config: dict[str, Any]) -> str:
    dataset_name = str(config["dataset"].get("name", ""))
    if dataset_name in DATASET_PRIMARY_METRICS:
        return DATASET_PRIMARY_METRICS[dataset_name]
    return str(config["dataset"].get("primary_metric", config["dataset"].get("metric", "accuracy")))


def metrics_from_probability(
    y: np.ndarray, probability: np.ndarray, *, threshold: float = 0.5
) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if probability.ndim == 1 or probability.shape[1] == 2:
        positive = probability if probability.ndim == 1 else probability[:, 1]
        prediction = (positive >= threshold).astype(int)
    else:
        positive = None
        prediction = probability.argmax(axis=1).astype(int)
    labels = sorted(set(y.tolist()) | set(prediction.tolist()))
    result: dict[str, Any] = {
        "n": int(len(y)),
        "threshold": float(threshold) if positive is not None else None,
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
    }
    if positive is not None and len(np.unique(y)) == 2:
        result["positive_f1"] = float(f1_score(y, prediction, pos_label=1, zero_division=0))
        result["roc_auc"] = float(roc_auc_score(y, positive))
    elif probability.ndim == 2 and len(np.unique(y)) > 2:
        try:
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(y, probability, multi_class="ovr", average="macro")
            )
        except ValueError:
            pass
    return result


def score(metrics: dict[str, Any], primary: str) -> float:
    aliases = {"macro_f1": "f1_macro", "auc": "roc_auc"}
    name = aliases.get(primary, primary)
    if name not in metrics:
        raise KeyError(f"Metric {primary!r} is unavailable in {sorted(metrics)}")
    return float(metrics[name])


def select_threshold(
    y: np.ndarray, probability: np.ndarray, primary: str
) -> tuple[float, dict[str, Any]]:
    if probability.ndim != 1 and probability.shape[1] != 2:
        metrics = metrics_from_probability(y, probability)
        return 0.5, metrics
    positive = probability if probability.ndim == 1 else probability[:, 1]
    if primary in {"roc_auc", "auc"}:
        return 0.5, metrics_from_probability(y, positive, threshold=0.5)
    candidates = np.unique(
        np.r_[0.05, 0.5, 0.95, np.quantile(positive, np.linspace(0.02, 0.98, 97))]
    )
    rows = [metrics_from_probability(y, positive, threshold=float(value)) for value in candidates]
    selected = max(
        rows,
        key=lambda row: (
            score(row, primary),
            row["balanced_accuracy"],
            -abs(float(row["threshold"]) - 0.5),
        ),
    )
    return float(selected["threshold"]), selected


def residual_feature_ranking(
    features: np.ndarray, residuals: np.ndarray, names: list[str]
) -> list[str]:
    """Rank columns by absolute standardized covariance with OOF residuals."""
    x = np.asarray(features, dtype=float)
    r = np.asarray(residuals, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    x = x - x.mean(axis=0, keepdims=True)
    x_scale = np.sqrt(np.square(x).mean(axis=0))
    r = r - r.mean(axis=0, keepdims=True)
    r_scale = np.sqrt(np.square(r).mean(axis=0))
    denominator = x_scale[:, None] * r_scale[None, :]
    covariance = np.abs(x.T @ r / max(len(x), 1))
    association = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 1e-12,
    ).max(axis=1)
    return [
        name
        for name, _ in sorted(
            zip(names, association.tolist()), key=lambda item: (-item[1], item[0])
        )
    ]


def choose_smallest_near_best(
    rows: list[dict[str, Any]], *, primary: str, margin: float
) -> dict[str, Any]:
    best = max(float(row["validation_primary_score"]) for row in rows)
    eligible = [row for row in rows if best - float(row["validation_primary_score"]) <= margin]
    return min(
        eligible,
        key=lambda row: (
            int(row["n_cluster_features"]),
            -float(row["validation_primary_score"]),
            row["family"],
        ),
    )


def _xgb(config: dict[str, Any], params: dict[str, Any], seed: int):
    labels = int(config["dataset"]["num_labels"])
    objective = "binary:logistic" if labels == 2 else "multi:softprob"
    extra = {} if labels == 2 else {"num_class": labels}
    return XGBClassifier(
        **params,
        **extra,
        objective=objective,
        random_state=seed,
        n_jobs=2,
        tree_method="hist",
        verbosity=0,
        eval_metric="logloss",
    )


def _lightgbm(config: dict[str, Any], params: dict[str, Any], seed: int):
    from lightgbm import LGBMClassifier

    labels = int(config["dataset"]["num_labels"])
    extra = {} if labels == 2 else {"num_class": labels}
    return LGBMClassifier(
        **params,
        **extra,
        objective="binary" if labels == 2 else "multiclass",
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _catboost(config: dict[str, Any], params: dict[str, Any], seed: int):
    from catboost import CatBoostClassifier

    labels = int(config["dataset"]["num_labels"])
    return CatBoostClassifier(
        **params,
        loss_function="Logloss" if labels == 2 else "MultiClass",
        random_seed=seed,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )


def model_candidates(config: dict[str, Any]) -> list[tuple[str, Callable, dict[str, Any]]]:
    rows: list[tuple[str, Callable, dict[str, Any]]] = [
        (
            "xgboost",
            _xgb,
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": 3,
                "reg_lambda": 1.0,
            },
        ),
        (
            "xgboost",
            _xgb,
            {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.03,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "min_child_weight": 8,
                "reg_alpha": 0.1,
                "reg_lambda": 3.0,
            },
        ),
    ]
    try:
        import lightgbm  # noqa: F401

        rows.extend(
            [
                (
                    "lightgbm",
                    _lightgbm,
                    {
                        "n_estimators": 400,
                        "num_leaves": 15,
                        "learning_rate": 0.04,
                        "min_child_samples": 30,
                        "colsample_bytree": 0.85,
                        "reg_lambda": 1.0,
                    },
                ),
                (
                    "lightgbm",
                    _lightgbm,
                    {
                        "n_estimators": 700,
                        "num_leaves": 31,
                        "learning_rate": 0.025,
                        "min_child_samples": 50,
                        "colsample_bytree": 0.75,
                        "reg_lambda": 3.0,
                    },
                ),
            ]
        )
    except ImportError:
        pass
    try:
        import catboost  # noqa: F401

        rows.extend(
            [
                (
                    "catboost",
                    _catboost,
                    {"iterations": 400, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
                ),
                (
                    "catboost",
                    _catboost,
                    {"iterations": 700, "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 7.0},
                ),
            ]
        )
    except ImportError:
        pass
    return rows


def _probability(model: Any, values: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(values), dtype=float)


def _oof_residuals(
    config: dict[str, Any],
    factory: Callable,
    params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    counts = np.bincount(y)
    folds = max(2, min(5, int(counts[counts > 0].min())))
    estimator = factory(config, params, 17)
    probability = cross_val_predict(
        estimator,
        x,
        y,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=137),
        method="predict_proba",
        n_jobs=1,
    )
    classes = probability.shape[1]
    truth = np.eye(classes, dtype=float)[y]
    return truth - probability


def _merge_pack(base: dict[str, Any], cot: dict[str, Any]) -> dict[str, Any]:
    return {
        **{
            split: merge_feature_frames(base[split], cot[split])
            for split in ("train", "val", "test")
        },
        "columns": [*base["columns"], *cot["columns"]],
    }


def run_cell(
    *,
    cell: Path,
    config: dict[str, Any],
    candidates_k: tuple[int, ...],
    tie_margin: float,
    output_name: str = "incremental_cluster_retuning_v1",
    cleanup_variant: str = "current",
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["output"]["base_dir"] = str(cell)
    config.setdefault("input", {})["cot_features_base_dir"] = str(cell)
    packs = build_feature_sets(config, [*BASE_FEATURES, "cot"])
    cluster_payload = json.loads((cell / "cot_clusters.json").read_text(encoding="utf-8"))
    clean_names, cleanup_report = select_clean_cluster_features(
        packs["cot"]["train"],
        cluster_payload.get("cluster_meta", []),
        variant=cleanup_variant,
    )
    if not clean_names and cleanup_variant != "current":
        raise ValueError(f"Cleanup variant {cleanup_variant} removed every cluster")
    packs["cot"] = {
        **{
            split: packs["cot"][split][["customer_id", "label", *clean_names]].copy()
            for split in ("train", "val", "test")
        },
        "columns": clean_names,
    }
    primary = metric_name(config)
    print(
        f"[incremental-retune] cell={cell} primary={primary} stage=base-selection",
        flush=True,
    )
    output = cell / output_name / "metrics.json"
    signature = fingerprint(
        {
            "protocol_version": PROTOCOL_VERSION,
            "cell": str(cell),
            "primary": primary,
            "candidates_k": list(candidates_k),
            "tie_margin": tie_margin,
            "cot_columns": packs["cot"]["columns"],
            "cleanup_variant": cleanup_variant,
            "cleanup_report": cleanup_report,
        }
    )
    if output.is_file():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old.get("artifact_signature") == signature:
            return old

    model_rows = model_candidates(config)
    base_sweep: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str, str], tuple[Any, np.ndarray]] = {}
    for base_name in BASE_FEATURES:
        pack = packs[base_name]
        x_train, y_train = split_xy(pack["train"], pack["columns"])
        x_val, y_val = split_xy(pack["val"], pack["columns"])
        for family, factory, params in model_rows:
            model = factory(config, params, 17)
            model.fit(x_train, y_train)
            probability = _probability(model, x_val)
            threshold, metrics = select_threshold(y_val, probability, primary)
            row = {
                "base_feature_set": base_name,
                "family": family,
                "params": params,
                "threshold": threshold,
                "validation_metrics": metrics,
                "validation_primary_score": score(metrics, primary),
            }
            base_sweep.append(row)
            fitted[(base_name, family, json.dumps(params, sort_keys=True))] = (model, probability)
    selected_base = max(
        base_sweep,
        key=lambda row: (
            row["validation_primary_score"],
            row["validation_metrics"]["balanced_accuracy"],
        ),
    )
    print(
        "[incremental-retune] "
        f"cell={cell} selected_base={selected_base['base_feature_set']} "
        f"family={selected_base['family']} "
        f"val={selected_base['validation_primary_score']:.6f} "
        "stage=residual-ranking",
        flush=True,
    )
    base_name = selected_base["base_feature_set"]
    base = packs[base_name]
    cot = packs["cot"]
    x_train_base, y_train = split_xy(base["train"], base["columns"])
    x_val_base, y_val = split_xy(base["val"], base["columns"])
    selected_factory = next(
        factory
        for family, factory, params in model_rows
        if family == selected_base["family"] and params == selected_base["params"]
    )
    residuals = _oof_residuals(
        config, selected_factory, selected_base["params"], x_train_base, y_train
    )
    x_train_cot, _ = split_xy(cot["train"], cot["columns"])
    ranking = residual_feature_ranking(x_train_cot, residuals, cot["columns"])
    combined = _merge_pack(base, cot)
    available_k = sorted({min(int(k), len(ranking)) for k in candidates_k})
    print(
        f"[incremental-retune] cell={cell} stage=augmented-sweep k={available_k}",
        flush=True,
    )
    augmented_sweep: list[dict[str, Any]] = []
    validation_probabilities: dict[str, np.ndarray] = {}
    for k in available_k:
        columns = [*base["columns"], *ranking[:k]]
        x_train, _ = split_xy(combined["train"], columns)
        x_val, _ = split_xy(combined["val"], columns)
        for family, factory, params in model_rows:
            model = factory(config, params, 17)
            model.fit(x_train, y_train)
            probability = _probability(model, x_val)
            threshold, metrics = select_threshold(y_val, probability, primary)
            key = fingerprint({"k": k, "family": family, "params": params})
            validation_probabilities[key] = probability
            augmented_sweep.append(
                {
                    "candidate_key": key,
                    "n_cluster_features": k,
                    "family": family,
                    "params": params,
                    "threshold": threshold,
                    "validation_metrics": metrics,
                    "validation_primary_score": score(metrics, primary),
                }
            )
    selected = choose_smallest_near_best(augmented_sweep, primary=primary, margin=tie_margin)

    base_key = (
        selected_base["base_feature_set"],
        selected_base["family"],
        json.dumps(selected_base["params"], sort_keys=True),
    )
    base_val_probability = fitted[base_key][1]
    augmented_val_probability = validation_probabilities[selected["candidate_key"]]
    blend_sweep = []
    for weight in np.linspace(0.0, 1.0, 21):
        probability = weight * augmented_val_probability + (1.0 - weight) * base_val_probability
        threshold, metrics = select_threshold(y_val, probability, primary)
        blend_sweep.append(
            {
                "augmented_weight": float(weight),
                "threshold": threshold,
                "validation_metrics": metrics,
                "validation_primary_score": score(metrics, primary),
            }
        )
    selected_blend = max(
        blend_sweep,
        key=lambda row: (row["validation_primary_score"], -abs(row["augmented_weight"] - 0.5)),
    )

    selected_columns = [*base["columns"], *ranking[: int(selected["n_cluster_features"])]]
    x_train_aug, _ = split_xy(combined["train"], selected_columns)
    x_test_aug, y_test = split_xy(combined["test"], selected_columns)
    x_test_base, _ = split_xy(base["test"], base["columns"])
    base_factory = next(
        factory
        for family, factory, params in model_rows
        if family == selected_base["family"] and params == selected_base["params"]
    )
    aug_factory = next(
        factory
        for family, factory, params in model_rows
        if family == selected["family"] and params == selected["params"]
    )
    runs = {}
    for seed in SEEDS:
        base_model = base_factory(config, selected_base["params"], seed)
        aug_model = aug_factory(config, selected["params"], seed)
        base_model.fit(x_train_base, y_train)
        aug_model.fit(x_train_aug, y_train)
        base_probability = _probability(base_model, x_test_base)
        augmented_probability = _probability(aug_model, x_test_aug)
        blended = (
            selected_blend["augmented_weight"] * augmented_probability
            + (1.0 - selected_blend["augmented_weight"]) * base_probability
        )
        runs[str(seed)] = {
            "base": metrics_from_probability(
                y_test, base_probability, threshold=selected_base["threshold"]
            ),
            "augmented": metrics_from_probability(
                y_test, augmented_probability, threshold=selected["threshold"]
            ),
            "blended": metrics_from_probability(
                y_test, blended, threshold=selected_blend["threshold"]
            ),
        }
    payload = {
        "artifact_signature": signature,
        "protocol": {
            "version": PROTOCOL_VERSION,
            "selection_split": "validation_only",
            "cluster_ranking": "train OOF residual association",
            "primary_metric": primary,
            "k_zero_control": True,
            "models_kept_separate": True,
            "cleanup_variant": cleanup_variant,
            "cleanup_is_train_only": True,
        },
        "cleanup_report": cleanup_report,
        "selected_base": selected_base,
        "residual_cluster_ranking": ranking,
        "augmented_sweep": augmented_sweep,
        "selected_augmented": selected,
        "blend_sweep": blend_sweep,
        "selected_blend": selected_blend,
        "test_runs": runs,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output, payload)
    print(
        "[incremental-retune] "
        f"cell={cell} completed selected_k={selected['n_cluster_features']} "
        f"family={selected['family']} "
        f"blend_weight={selected_blend['augmented_weight']:.2f}",
        flush=True,
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, type=Path, help="Output of ife select-features")
    parser.add_argument("--cluster-counts", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--tie-margin", type=float, default=0.002)
    parser.add_argument("--cleanup", choices=CLEANUP_VARIANTS, default="current")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if 0 not in args.cluster_counts or min(args.cluster_counts) < 0:
        parser.error("Cluster counts must be nonnegative and include the zero-claim control")
    print(
        json.dumps(
            {
                "cell": str(args.cell),
                "cluster_counts": args.cluster_counts,
                "cleanup": args.cleanup,
                "mode": "execute" if args.execute else "dry-run",
            },
            indent=2,
        )
    )
    if args.execute:
        config = json.loads((args.cell / "source_manifest.json").read_text())["source_manifest"][
            "config"
        ]
        run_cell(
            cell=args.cell,
            config=config,
            candidates_k=tuple(sorted(set(args.cluster_counts))),
            tie_margin=args.tie_margin,
            cleanup_variant=args.cleanup,
            output_name=f"incremental_{args.cleanup}",
        )


if __name__ == "__main__":
    main()

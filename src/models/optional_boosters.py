"""Validation-selected LightGBM/CatBoost evaluation for v5 folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.models.ml_baseline import (
    build_feature_sets,
    evaluate,
    primary_score,
    split_xy,
)


def _mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _select(
    factory: Callable[[dict[str, Any], int], Any],
    candidates: list[dict[str, Any]],
    *,
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x_train, y_train = train
    x_validation, y_validation = validation
    rows = []
    for params in candidates:
        model = factory(params, 17)
        model.fit(x_train, y_train)
        metrics = evaluate(model, x_validation, y_validation)
        rows.append({
            "params": params,
            "validation_metrics": metrics,
            "selection_score": primary_score(metrics, config),
        })
    selected = max(
        rows,
        key=lambda row: (
            row["selection_score"],
            -len(json.dumps(row["params"], sort_keys=True)),
        ),
    )
    return selected["params"], rows


def _evaluate_seeds(
    factory: Callable[[dict[str, Any], int], Any],
    params: dict[str, Any],
    *,
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    seeds: list[int],
) -> dict[str, Any]:
    x_train, y_train = train
    x_validation, y_validation = validation
    x_test, y_test = test
    runs = {}
    for seed in seeds:
        model = factory(params, seed)
        model.fit(
            np.concatenate([x_train, x_validation]),
            np.concatenate([y_train, y_validation]),
        )
        runs[str(seed)] = evaluate(model, x_test, y_test)
    metric_names = sorted({
        name for run in runs.values() for name, value in run.items()
        if isinstance(value, (int, float)) and name != "n"
    })
    return {
        "params": params,
        "runs": runs,
        "test_summary": {
            metric: _mean_sd(
                [float(run[metric]) for run in runs.values() if metric in run]
            )
            for metric in metric_names
        },
    }


def run_optional_boosters(
    config: dict[str, Any],
    *,
    experiments: list[str],
    output_path: Path,
) -> dict[str, Any]:
    """Run available boosters and explicitly report missing optional packages."""
    feature_sets = build_feature_sets(config, experiments)
    seeds = [int(value) for value in config["evaluation"]["seeds"]]
    implementations: dict[
        str, tuple[Callable[[dict[str, Any], int], Any], list[dict[str, Any]]]
    ] = {}
    unavailable = {}
    try:
        from lightgbm import LGBMClassifier

        def lightgbm_factory(params: dict[str, Any], seed: int):
            return LGBMClassifier(
                **params,
                random_state=seed,
                n_jobs=2,
                verbosity=-1,
            )

        implementations["lightgbm"] = (
            lightgbm_factory,
            [
                {"n_estimators": 200, "num_leaves": 15,
                 "learning_rate": 0.07, "class_weight": None},
                {"n_estimators": 500, "num_leaves": 31,
                 "learning_rate": 0.03, "class_weight": None},
                {"n_estimators": 200, "num_leaves": 15,
                 "learning_rate": 0.07, "class_weight": "balanced"},
                {"n_estimators": 500, "num_leaves": 31,
                 "learning_rate": 0.03, "class_weight": "balanced"},
            ],
        )
    except ImportError:
        unavailable["lightgbm"] = "Install lightgbm to run this evaluator"
    try:
        from catboost import CatBoostClassifier

        def catboost_factory(params: dict[str, Any], seed: int):
            return CatBoostClassifier(
                **params,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=2,
            )

        implementations["catboost"] = (
            catboost_factory,
            [
                {"iterations": 200, "depth": 4, "learning_rate": 0.07},
                {"iterations": 500, "depth": 6, "learning_rate": 0.03},
                {"iterations": 200, "depth": 4, "learning_rate": 0.07,
                 "auto_class_weights": "Balanced"},
                {"iterations": 500, "depth": 6, "learning_rate": 0.03,
                 "auto_class_weights": "Balanced"},
            ],
        )
    except ImportError:
        unavailable["catboost"] = "Install catboost to run this evaluator"

    payload: dict[str, Any] = {
        "selection_split": "inner_validation",
        "final_fit": "inner_train_plus_inner_validation",
        "primary_metric": config["dataset"]["primary_metric"],
        "seeds": seeds,
        "unavailable": unavailable,
        "feature_sets": {},
    }
    for feature_name, pack in feature_sets.items():
        columns = pack["columns"]
        arrays = {
            split: split_xy(pack[split], columns)
            for split in ("train", "val", "test")
        }
        payload["feature_sets"][feature_name] = {}
        for implementation, (factory, candidates) in implementations.items():
            selected, sweep = _select(
                factory,
                candidates,
                train=arrays["train"],
                validation=arrays["val"],
                config=config,
            )
            payload["feature_sets"][feature_name][implementation] = {
                "validation_sweep": sweep,
                **_evaluate_seeds(
                    factory,
                    selected,
                    train=arrays["train"],
                    validation=arrays["val"],
                    test=arrays["test"],
                    seeds=seeds,
                ),
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return payload

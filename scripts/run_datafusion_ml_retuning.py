#!/usr/bin/env python3
"""Retune Data Fusion classifiers on frozen v11 feature matrices.

All model/threshold choices use validation only. Test labels are evaluated only
after a family configuration and threshold have been frozen. Existing ML
artifacts are never modified.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
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
    roc_curve,
)
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v4_offline_pipeline import load_source
from src.experiments.artifacts import atomic_write_json, files_fingerprint, fingerprint
from src.models.ml_baseline import build_feature_sets, split_xy


MODELS = ("qwen", "gpt_oss")
DEFAULT_FEATURES = (
    "standard", "handcrafted", "all_nonclaim", "cot", "concat", "all_features",
)
SEEDS = (17, 101, 947)
PROTOCOL_VERSION = 1


def probability_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (np.asarray(probability) >= float(threshold)).astype(int)
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def select_balanced_threshold(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Choose threshold on validation, preferring the value closest to 0.5."""
    _, _, thresholds = roc_curve(y, probability)
    candidates = sorted({0.0, 0.5, 1.0, *[
        float(value) for value in thresholds if np.isfinite(value)
    ]})
    rows = [probability_metrics(y, probability, threshold) for threshold in candidates]
    selected = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["positive_f1"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return {
        "threshold": float(selected["threshold"]),
        "validation_balanced_accuracy": float(selected["balanced_accuracy"]),
        "validation_positive_f1": float(selected["positive_f1"]),
    }


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def family_summary(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = sorted({
        key for run in runs.values() for key, value in run["test"].items()
        if isinstance(value, (int, float)) and key not in {"n", "threshold"}
    })
    return {
        key: mean_sd([float(run["test"][key]) for run in runs.values()])
        for key in metrics
    }


def xgboost_factory(parameters: dict[str, Any], seed: int):
    return XGBClassifier(
        **parameters,
        objective="binary:logistic",
        random_state=seed,
        n_jobs=2,
        verbosity=0,
        eval_metric="auc",
        tree_method="hist",
    )


def lightgbm_factory(parameters: dict[str, Any], seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        **parameters,
        objective="binary",
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def catboost_factory(parameters: dict[str, Any], seed: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        **parameters,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def candidates(
    *, existing_xgb: dict[str, Any], positive_ratio: float,
) -> dict[str, tuple[Callable[[dict[str, Any], int], Any], list[dict[str, Any]]]]:
    weights = [1.0, math.sqrt(positive_ratio), positive_ratio]
    xgb = [{**existing_xgb, "scale_pos_weight": float(weight)} for weight in weights]
    lightgbm = []
    for estimators, leaves, minimum_leaf in ((300, 15, 20), (600, 31, 20), (900, 31, 50)):
        for weight in (None, "balanced"):
            lightgbm.append({
                "n_estimators": estimators,
                "num_leaves": leaves,
                "max_depth": -1,
                "learning_rate": 0.03,
                "min_child_samples": minimum_leaf,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "class_weight": weight,
            })
    catboost = []
    for iterations, depth in ((300, 4), (600, 6), (900, 7)):
        for balanced in (False, True):
            row: dict[str, Any] = {
                "iterations": iterations,
                "depth": depth,
                "learning_rate": 0.03,
                "l2_leaf_reg": 3.0,
            }
            if balanced:
                row["auto_class_weights"] = "Balanced"
            catboost.append(row)
    return {
        "xgboost": (xgboost_factory, xgb),
        "lightgbm": (lightgbm_factory, lightgbm),
        "catboost": (catboost_factory, catboost),
    }


def select_family(
    factory: Callable[[dict[str, Any], int], Any],
    parameter_grid: list[dict[str, Any]],
    *, x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    rows = []
    for parameters in parameter_grid:
        model = factory(parameters, SEEDS[0])
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_val)[:, 1]
        threshold = select_balanced_threshold(y_val, probability)
        rows.append({
            "params": parameters,
            "validation_roc_auc": float(roc_auc_score(y_val, probability)),
            **threshold,
        })
    selected = max(
        rows,
        key=lambda row: (
            row["validation_roc_auc"],
            row["validation_balanced_accuracy"],
            -len(json.dumps(row["params"], sort_keys=True)),
        ),
    )
    return selected["params"], {
        "threshold": selected["threshold"],
        "validation_roc_auc": selected["validation_roc_auc"],
        "validation_balanced_accuracy": selected["validation_balanced_accuracy"],
        "validation_positive_f1": selected["validation_positive_f1"],
    }, rows


def run_cell(cell: Path, features: tuple[str, ...]) -> dict[str, Any]:
    source = json.loads((cell / "source_manifest.json").read_text(encoding="utf-8"))[
        "source_contract"
    ]
    _, config = load_source(Path(source["source_root"]))
    derived = copy.deepcopy(config)
    derived["output"]["base_dir"] = str(cell)
    derived.setdefault("input", {})["cot_features_base_dir"] = str(cell)
    packs = build_feature_sets(derived, list(features))
    existing = json.loads((cell / "ml_metrics.json").read_text(encoding="utf-8"))
    output_dir = cell / "ml_retuned_v1"
    output_path = output_dir / "metrics.json"
    signature = fingerprint({
        "protocol_version": PROTOCOL_VERSION,
        "splits": files_fingerprint(
            [Path(path) for path in derived["dataset"]["splits"].values()]
        ),
        "cot_features": files_fingerprint([
            cell / f"cot_features_{split}.parquet"
            for split in ("train", "val", "test")
        ]),
        "features": list(features),
        "seeds": list(SEEDS),
    })
    payload: dict[str, Any] = {
        "artifact_signature": signature,
        "protocol": {
            "version": PROTOCOL_VERSION,
            "selection": "validation ROC-AUC; threshold by validation balanced accuracy",
            "test_access": "after family parameters and threshold are frozen",
            "fit": "train only to keep validation-derived threshold calibrated",
            "seeds": list(SEEDS),
        },
        "cell": str(cell),
        "features": {},
    }
    if output_path.is_file():
        old = json.loads(output_path.read_text(encoding="utf-8"))
        if old.get("artifact_signature") == signature:
            payload["features"].update(old.get("features", {}))
    for feature_name in features:
        pack = packs[feature_name]
        columns = pack["columns"]
        x_train, y_train = split_xy(pack["train"], columns)
        x_val, y_val = split_xy(pack["val"], columns)
        x_test, y_test = split_xy(pack["test"], columns)
        negative, positive = np.bincount(y_train, minlength=2)
        ratio = float(negative / max(positive, 1))
        grids = candidates(
            existing_xgb=existing[feature_name]["xgboost"]["params"],
            positive_ratio=ratio,
        )
        payload["features"].setdefault(feature_name, {})
        for family, (factory, grid) in grids.items():
            if family in payload["features"][feature_name]:
                continue
            params, selection, sweep = select_family(
                factory, grid,
                x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
            )
            threshold = float(selection["threshold"])
            runs = {}
            for seed in SEEDS:
                model = factory(params, seed)
                model.fit(x_train, y_train)
                val_probability = model.predict_proba(x_val)[:, 1]
                test_probability = model.predict_proba(x_test)[:, 1]
                runs[str(seed)] = {
                    "val": probability_metrics(y_val, val_probability, threshold),
                    "test": probability_metrics(y_test, test_probability, threshold),
                }
            payload["features"][feature_name][family] = {
                "selected_params": params,
                "selection": selection,
                "validation_sweep": sweep,
                "runs": runs,
                "test_summary": family_summary(runs),
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output_path, payload)
            print(f"completed {cell.parent.name}/{feature_name}/{family}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-root", type=Path,
        default=Path("results/isolated/derived/reviewer-v11-isolated-r2"),
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--features", nargs="+", choices=DEFAULT_FEATURES, default=list(DEFAULT_FEATURES))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cells = [
        args.derived_root / "datafusion_default_2023" / model / "seed_17"
        for model in args.models
    ]
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "cells": [str(cell) for cell in cells],
        "features": args.features,
        "families": ["xgboost", "lightgbm", "catboost"],
        "selection": "validation only",
        "outputs": [str(cell / "ml_retuned_v1/metrics.json") for cell in cells],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    for cell in cells:
        run_cell(cell, tuple(args.features))


if __name__ == "__main__":
    main()

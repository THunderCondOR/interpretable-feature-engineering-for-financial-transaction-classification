#!/usr/bin/env python3
"""Evaluate frozen v4 features with the PTLS LightGBM protocol."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v4_offline_pipeline import load_source
from src.experiments.artifacts import atomic_write_json
from src.models.ml_baseline import (
    _seed_metric_summary,
    build_feature_sets,
    evaluate,
    merge_feature_frames,
    split_xy,
)


CELLS = tuple(
    (dataset, model)
    for model in ("qwen", "gpt_oss")
    for dataset in ("rosbank", "gender", "age")
)


def ptls_parameters(dataset: str) -> dict:
    common = {
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "subsample_freq": 1,
        "feature_fraction": 0.75,
        "lambda_l1": 1,
        "lambda_l2": 1,
        "min_data_in_leaf": 50,
        "random_state": 42,
        "n_jobs": 8,
        "verbosity": -1,
    }
    if dataset == "age":
        return {
            **common,
            "n_estimators": 1000,
            "objective": "multiclass",
            "num_class": 4,
            "metric": "multi_error",
            "subsample": 0.75,
            "max_depth": 12,
            "num_leaves": 50,
            "n_jobs": 4,
        }
    return {
        **common,
        "n_estimators": 500,
        "objective": "binary",
        "metric": "auc",
        "subsample": 0.5,
        "max_depth": 6,
    }


def prefixed_pack_for_lightgbm(pack: dict, prefix: str) -> dict:
    columns = list(pack["columns"])
    mapping = {column: f"{prefix}{column}" for column in columns}
    return {
        split: pack[split].rename(columns=mapping)
        for split in ("train", "val", "test")
    } | {"columns": [mapping[column] for column in columns]}


def feature_packs(config: dict) -> dict[str, dict]:
    packs = build_feature_sets(
        config,
        [
            "standard",
            "llm_profile",
            "standard_profile",
            "handcrafted",
            "cot",
            "concat",
        ],
    )
    standard_cot = {
        split: merge_feature_frames(
            packs["standard"][split], packs["cot"][split]
        )
        for split in ("train", "val", "test")
    }
    standard_cot["columns"] = [
        column
        for column in standard_cot["train"].columns
        if column not in {"customer_id", "label"}
    ]
    standard_profile_cot = {
        split: merge_feature_frames(
            packs["standard_profile"][split], packs["cot"][split]
        )
        for split in ("train", "val", "test")
    }
    standard_profile_cot["columns"] = [
        column
        for column in standard_profile_cot["train"].columns
        if column not in {"customer_id", "label"}
    ]
    # Standard and handcrafted contain overlapping generic names. Prefix both
    # namespaces before combining them so no signal is silently discarded.
    renamed = {
        name: prefixed_pack_for_lightgbm(packs[name], prefix)
        for name, prefix in (
            ("standard", "std__"),
            ("handcrafted", "hc__"),
        )
    }
    all_features = {}
    for split in ("train", "val", "test"):
        base = merge_feature_frames(
            renamed["standard"][split], renamed["handcrafted"][split]
        )
        all_features[split] = merge_feature_frames(base, packs["cot"][split])
    all_features["columns"] = [
        column
        for column in all_features["train"].columns
        if column not in {"customer_id", "label"}
    ]
    standard_profile = prefixed_pack_for_lightgbm(
        packs["standard_profile"], "stdp__"
    )
    handcrafted = prefixed_pack_for_lightgbm(
        packs["handcrafted"], "hc__"
    )
    all_profile = {}
    for split in ("train", "val", "test"):
        base = merge_feature_frames(
            standard_profile[split], handcrafted[split]
        )
        all_profile[split] = merge_feature_frames(base, packs["cot"][split])
    all_profile["columns"] = [
        column
        for column in all_profile["train"].columns
        if column not in {"customer_id", "label"}
    ]
    return {
        **packs,
        "standard_cot": standard_cot,
        "standard_profile_cot": standard_profile_cot,
        "all": all_features,
        "all_profile": all_profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("results/v2/derived/reviewer-v4-offline-v2"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "protocol": "PTLS fixed LightGBM parameters",
        "cv_folds": args.folds,
        "random_state": 42,
        "feature_sets": [
            "standard", "llm_profile", "standard_profile",
            "handcrafted", "cot", "concat", "standard_cot",
            "standard_profile_cot", "all", "all_profile",
        ],
        "cells": [
            str(args.derived_root / dataset / model / "seed_17")
            for dataset, model in CELLS
        ],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise RuntimeError(
            "LightGBM is optional and not installed. Install the dependencies "
            "from requirements-lightgbm.txt before using --execute."
        ) from error

    for dataset, model_slug in CELLS:
        cell = args.derived_root / dataset / model_slug / "seed_17"
        source = json.loads(
            (cell / "source_manifest.json").read_text(encoding="utf-8")
        )["source_contract"]
        _, config = load_source(Path(source["source_root"]))
        derived = copy.deepcopy(config)
        derived["output"]["base_dir"] = str(cell)
        derived.setdefault("input", {})["cot_features_base_dir"] = str(cell)
        results = {}
        for feature_set, pack in feature_packs(derived).items():
            development = pd.concat(
                [pack["train"], pack["val"]], ignore_index=True
            )
            columns = pack["columns"]
            values, labels = split_xy(development, columns)
            test_values, test_labels = split_xy(pack["test"], columns)
            folds = StratifiedKFold(
                n_splits=args.folds, shuffle=True, random_state=42
            )
            runs = {}
            for fold, (train_index, validation_index) in enumerate(
                folds.split(values, labels)
            ):
                classifier = LGBMClassifier(**ptls_parameters(dataset))
                classifier.fit(values[train_index], labels[train_index])
                runs[str(fold)] = {
                    "val": evaluate(
                        classifier,
                        values[validation_index],
                        labels[validation_index],
                    ),
                    "test": evaluate(
                        classifier, test_values, test_labels
                    ),
                }
            results[feature_set] = {
                "params": ptls_parameters(dataset),
                "runs": runs,
                "summary": {
                    split: _seed_metric_summary(runs, split)
                    for split in ("val", "test")
                },
            }
            print(
                f"{dataset}/{model_slug}/{feature_set}: completed "
                f"{args.folds} folds",
                flush=True,
            )
        output = cell / "lightgbm_ptls"
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / "metrics.json", {
            "protocol": plan["protocol"],
            "cv_folds": args.folds,
            "random_state": 42,
            "dataset": dataset,
            "model": model_slug,
            "results": results,
        })


if __name__ == "__main__":
    main()

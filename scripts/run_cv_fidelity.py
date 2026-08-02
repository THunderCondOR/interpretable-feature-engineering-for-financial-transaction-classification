#!/usr/bin/env python3
"""Run fold-pure teacher/surrogate fidelity for the CV benchmarks.

Unlike the legacy fidelity suite, this runner never merges claim spaces from
different generators.  A non-claim teacher is selected on each fold's inner
validation split and each LLM's frozen claim representation is evaluated
separately on the outer test split.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_fidelity_suite import (
    NONCLAIM,
    aligned_probabilities,
    fit_teacher_model,
    model_specs,
    reusable_teacher_selection,
)
from src.data.benchmark_registry import BENCHMARKS
from src.experiments.artifacts import atomic_write_json, files_fingerprint
from src.experiments.config_builder import load_yaml
from src.models.ml_baseline import (
    build_feature_sets,
    evaluate,
    primary_score,
    split_xy,
)


DEFAULT_DATASET = "berka"
DEFAULT_PROTOCOL = "unittab_70_30_5seed"
DEFAULT_MODELS = ("qwen", "gpt_oss")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
METRIC_KEYS = (
    "hard_agreement",
    "probability_mae",
    "probability_rmse",
    "jensen_shannon_divergence",
    "agree_and_correct",
    "agree_and_wrong",
    "teacher_only_correct",
    "surrogate_only_correct",
)


def cv_config(
    *,
    dataset: str,
    protocol: str,
    fold: int,
    generated_root: Path,
    prepared_root: Path,
    derived_cell: Path,
) -> dict[str, Any]:
    """Build the exact inner-train/validation/outer-test ML configuration."""
    selected = generated_root / dataset / f"fold_{fold}" / "selected_qwen.yaml"
    if not selected.is_file():
        raise FileNotFoundError(f"Missing selected fold config: {selected}")
    config = copy.deepcopy(load_yaml(selected))
    fold_root = prepared_root / dataset / protocol / f"fold_{fold}"
    events = prepared_root / dataset / protocol / "events.parquet"
    required = [
        events,
        fold_root / "inner_train_ids.json",
        fold_root / "inner_validation_ids.json",
        fold_root / "outer_test_ids.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CV inputs: {missing}")
    config["dataset"]["splits"] = {
        "train": str(events), "val": str(events), "test": str(events),
    }
    config["dataset"]["client_ids_by_split"] = {
        "train": str(fold_root / "inner_train_ids.json"),
        "val": str(fold_root / "inner_validation_ids.json"),
        "test": str(fold_root / "outer_test_ids.json"),
    }
    config["output"]["base_dir"] = str(derived_cell)
    config.setdefault("input", {})["cot_features_base_dir"] = str(derived_cell)
    return config


def _write_teacher_predictions(
    *,
    model: Any,
    pack: dict[str, Any],
    columns: list[str],
    n_classes: int,
    output_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        frame = pack[split].reset_index(drop=True)
        values, _ = split_xy(frame, columns)
        probabilities = aligned_probabilities(model, values, n_classes)
        records = frame[["customer_id", "label"]].copy()
        for label in range(n_classes):
            records[f"teacher_prob_{label}"] = probabilities[:, label]
        path = output_dir / f"teacher_predictions_{split}.csv"
        temporary = path.with_suffix(".csv.tmp")
        records.to_csv(temporary, index=False)
        temporary.replace(path)
        paths[split] = path
    return paths


def train_cv_teacher(config: dict[str, Any], output_dir: Path) -> Path:
    """Select a teacher on inner validation and keep validation out-of-fit."""
    selection_path = output_dir / "teacher_selection.json"
    if reusable_teacher_selection(selection_path):
        return selection_path
    packs = build_feature_sets(config, list(NONCLAIM))
    specs, unavailable = model_specs(config)
    n_classes = int(config["dataset"]["num_labels"])
    candidates: list[dict[str, Any]] = []
    for feature_set, pack in packs.items():
        columns = pack["columns"]
        x_train, y_train = split_xy(pack["train"], columns)
        x_val, y_val = split_xy(pack["val"], columns)
        for spec_index, spec in enumerate(specs):
            model = fit_teacher_model(
                spec, spec["factory"](17), x_train, y_train
            )
            probabilities = aligned_probabilities(model, x_val, n_classes)
            metrics = evaluate(model, x_val, y_val)
            candidates.append({
                "feature_set": feature_set,
                "spec_index": spec_index,
                "backend": spec["backend"],
                "size": spec["size"],
                "balanced": spec["balanced"],
                "primary_metric": config["dataset"]["primary_metric"],
                "primary_score": float(primary_score(metrics, config)),
                "log_loss": float(log_loss(
                    y_val, probabilities, labels=np.arange(n_classes)
                )),
                "validation_metrics": metrics,
            })
    if not candidates:
        raise RuntimeError("No usable teacher candidates")
    selected = sorted(candidates, key=lambda row: (
        -row["primary_score"], row["log_loss"], row["feature_set"],
        row["backend"], row["size"], row["balanced"],
    ))[0]
    pack = packs[selected["feature_set"]]
    columns = pack["columns"]
    x_train, y_train = split_xy(pack["train"], columns)
    spec = specs[int(selected["spec_index"])]
    model = fit_teacher_model(spec, spec["factory"](17), x_train, y_train)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _write_teacher_predictions(
        model=model,
        pack=pack,
        columns=columns,
        n_classes=n_classes,
        output_dir=output_dir,
    )
    payload = {
        "protocol": {
            "selection": "inner_train_to_inner_validation",
            "fit_population": "inner_train_only",
            "evaluation": "frozen_outer_test",
            "primary_metric": config["dataset"]["primary_metric"],
            "seed": 17,
        },
        "selected_teacher": f"{selected['feature_set']}:{selected['backend']}",
        "selected_candidate": selected,
        "candidate_metrics": candidates,
        "unavailable_backends": unavailable,
        "selected": {
            "paths": {key: str(path) for key, path in paths.items()},
            "filters": {},
            "file_hashes": files_fingerprint(paths.values()),
        },
    }
    atomic_write_json(selection_path, payload)
    return selection_path


def fidelity_cells(
    *, folds: tuple[int, ...], models: tuple[str, ...]
) -> list[tuple[int, str]]:
    return [(fold, model) for fold in folds for model in models]


def run_fidelity_cell(
    *,
    python_bin: Path,
    feature_cell: Path,
    teacher_selection: Path,
    output_dir: Path,
    bootstrap_samples: int,
    permutation_controls: int,
) -> None:
    command = [
        str(python_bin), "scripts/run_fidelity_analysis.py",
        "--train-features", str(feature_cell / "cot_features_train.parquet"),
        "--val-features", str(feature_cell / "cot_features_val.parquet"),
        "--test-features", str(feature_cell / "cot_features_test.parquet"),
        "--teacher-selection", str(teacher_selection),
        "--cluster-metadata", str(feature_cell / "cot_clusters.json"),
        "--output-dir", str(output_dir),
        "--feature-counts", "25", "50", "100",
        "--bootstrap-samples", str(bootstrap_samples),
        "--permutation-controls", str(permutation_controls),
        "--teacher-seed", "17",
        "--surrogate-seed", "17",
        "--execute",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def aggregate_results(output_root: Path, models: tuple[str, ...], folds: tuple[int, ...]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold, model in fidelity_cells(folds=folds, models=models):
        path = output_root / f"fold_{fold}" / model / "fidelity_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected_family = payload["surrogate_selection"]["selected_surrogate"]
        for family in ("logistic_regression", "xgboost", "shallow_tree"):
            metrics = payload["results"][family]["test"]
            row = {
                "fold": fold,
                "model": model,
                "surrogate": family,
                "selected_on_validation": family == selected_family,
            }
            row.update({key: float(metrics[key]) for key in METRIC_KEYS})
            rows.append(row)
        selected_metrics = payload["results"][selected_family]["test"]
        selected_row = {
            "fold": fold,
            "model": model,
            "surrogate": "validation_selected",
            "selected_family": selected_family,
            "selected_on_validation": True,
        }
        selected_row.update({key: float(selected_metrics[key]) for key in METRIC_KEYS})
        rows.append(selected_row)
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "fidelity_by_fold.csv", index=False)
    summaries = []
    for (model, surrogate), group in frame.groupby(["model", "surrogate"], sort=True):
        record: dict[str, Any] = {
            "model": model,
            "surrogate": surrogate,
            "n_folds": int(len(group)),
        }
        for key in METRIC_KEYS:
            values = group[key].to_numpy(float)
            record[key] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        summaries.append(record)
    payload = {
        "dataset": DEFAULT_DATASET,
        "protocol": DEFAULT_PROTOCOL,
        "claim_spaces": "separate_per_source_model",
        "folds": list(folds),
        "models": list(models),
        "summary": summaries,
    }
    atomic_write_json(output_root / "fidelity_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--models", default="qwen,gpt_oss")
    parser.add_argument(
        "--generated-root", type=Path,
        default=Path("logs/runs/reviewer-v5-fixed-new-datasets/generated"),
    )
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("data/benchmarks_v5")
    )
    parser.add_argument(
        "--derived-root", type=Path,
        default=Path("results/v5/derived/cv_main_e5"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/v5/derived/cv_fidelity/berka/unittab_70_30_5seed"),
    )
    parser.add_argument(
        "--python-bin", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutation-controls", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    folds = tuple(int(value) for value in args.folds.split(",") if value)
    models = tuple(value for value in args.models.split(",") if value)
    if args.dataset != DEFAULT_DATASET or args.protocol != DEFAULT_PROTOCOL:
        raise ValueError("This audited runner currently supports Berka unittab_70_30_5seed only")
    if args.dataset not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {args.dataset}")
    if not folds or set(folds) - set(DEFAULT_FOLDS):
        raise ValueError(f"Invalid folds: {folds}")
    if not models or set(models) - set(DEFAULT_MODELS):
        raise ValueError(f"Invalid models: {models}")
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "dataset": args.dataset,
        "protocol": args.protocol,
        "folds": list(folds),
        "models": list(models),
        "cells": len(folds) * len(models),
        "teacher": "one validation-selected non-claim teacher per fold",
        "claim_spaces": "Qwen and GPT-OSS remain separate; no union",
        "output_root": str(args.output_root),
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    for fold in folds:
        representative_cell = (
            args.derived_root / args.dataset / args.protocol / f"fold_{fold}" / models[0]
        )
        config = cv_config(
            dataset=args.dataset,
            protocol=args.protocol,
            fold=fold,
            generated_root=args.generated_root,
            prepared_root=args.prepared_root,
            derived_cell=representative_cell,
        )
        teacher = train_cv_teacher(
            config, args.output_root / f"fold_{fold}" / "teacher"
        )
        for model in models:
            feature_cell = (
                args.derived_root / args.dataset / args.protocol
                / f"fold_{fold}" / model
            )
            required = [
                feature_cell / f"cot_features_{split}.parquet"
                for split in ("train", "val", "test")
            ] + [feature_cell / "cot_clusters.json"]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing fidelity inputs: {missing}")
            run_fidelity_cell(
                python_bin=args.python_bin,
                feature_cell=feature_cell,
                teacher_selection=teacher,
                output_dir=args.output_root / f"fold_{fold}" / model,
                bootstrap_samples=args.bootstrap_samples,
                permutation_controls=args.permutation_controls,
            )
    aggregate_results(args.output_root, models, folds)
    print(f"Saved Berka CV fidelity -> {args.output_root}")


if __name__ == "__main__":
    main()

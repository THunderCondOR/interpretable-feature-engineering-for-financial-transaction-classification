#!/usr/bin/env python3
"""Select only incrementally useful cluster features for Concat and evaluate them."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v4_offline_pipeline import load_source
from src.experiments.artifacts import (
    atomic_write_json,
    files_fingerprint,
    fingerprint,
)
from src.models.ml_baseline import (
    _balanced_accuracy_ci,
    _seed_metric_summary,
    build_feature_sets,
    evaluate,
    merge_feature_frames,
    split_xy,
    tune_xgboost,
    xgb_objective,
)


CELLS = tuple(
    (dataset, model)
    for model in ("qwen", "gpt_oss")
    for dataset in ("rosbank", "gender", "age")
)
DEFAULT_K = (0, 5, 10, 20, 50, 100, 200)
MIXTURES = {
    "handcrafted": "selected_concat",
    "standard": "selected_standard_concat",
    "all": "selected_all_concat",
}


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def queue_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("state") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def prerequisites_complete(root: Path) -> bool:
    queues = [root / f"ml_queue_{model}.json" for model in ("qwen", "gpt_oss")]
    stability = [
        root / dataset / model / "seed_17" / "stability" / "cluster_seeds" / "summary.json"
        for dataset, model in CELLS
    ]
    return all(queue_complete(path) for path in queues) and all(
        path.is_file() for path in stability
    )


def choose_smallest_near_best(rows: list[dict], tie_margin: float) -> dict:
    best = max(float(row["validation_balanced_accuracy"]) for row in rows)
    eligible = [
        row
        for row in rows
        if float(row["validation_balanced_accuracy"]) >= best - tie_margin
    ]
    return min(eligible, key=lambda row: int(row["n_cluster_features"]))


def xgb(config: dict, params: dict, seed: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective=xgb_objective(config),
        random_state=seed,
        n_jobs=2,
        verbosity=0,
        eval_metric="logloss",
        tree_method="hist",
    )


def prefixed_pack(pack: dict, prefix: str) -> dict:
    columns = list(pack["columns"])
    rename = {column: f"{prefix}{column}" for column in columns}
    return {
        split: pack[split].rename(columns=rename)
        for split in ("train", "val", "test")
    } | {"columns": [rename[column] for column in columns]}


def build_base_pack(packs: dict, mixture: str) -> dict:
    if mixture in {"standard", "handcrafted"}:
        return packs[mixture]
    standard = prefixed_pack(packs["standard"], "std__")
    handcrafted = prefixed_pack(packs["handcrafted"], "hc__")
    frames = {
        split: merge_feature_frames(standard[split], handcrafted[split])
        for split in ("train", "val", "test")
    }
    return {
        **frames,
        "columns": [*standard["columns"], *handcrafted["columns"]],
    }


def run_cell(
    cell: Path,
    *,
    mixture: str,
    tie_margin: float,
    candidates: tuple[int, ...],
) -> dict:
    feature_set = MIXTURES[mixture]
    output = cell / feature_set
    metrics_path = output / "metrics.json"
    predictions_path = output / "predictions.jsonl"
    selection_path = cell / "cluster_selection.json"
    source = json.loads((cell / "source_manifest.json").read_text(encoding="utf-8"))[
        "source_contract"
    ]
    _, config = load_source(Path(source["source_root"]))
    derived = copy.deepcopy(config)
    derived["output"]["base_dir"] = str(cell)
    derived.setdefault("input", {})["cot_features_base_dir"] = str(cell)
    derived.setdefault("evaluation", {})["seeds"] = [17, 101, 947]
    derived.setdefault("optuna", {})["n_trials"] = 30

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    ordered = selection["selected_representation"]["selected_feature_names"]
    requested = ["cot", mixture] if mixture != "all" else [
        "standard", "handcrafted", "cot"
    ]
    packs = build_feature_sets(derived, requested)
    base, cot = build_base_pack(packs, mixture), packs["cot"]
    ordered = [name for name in ordered if name in cot["columns"]]
    available_k = sorted({min(value, len(ordered)) for value in candidates})
    source_files = [
        selection_path,
        *(cell / f"cot_features_{split}.parquet" for split in ("train", "val", "test")),
        *(Path(path) for path in derived["dataset"]["splits"].values()),
    ]
    signature = fingerprint({
        "stage": f"{feature_set}_v1",
        "inputs": files_fingerprint(source_files),
        "candidates": available_k,
        "tie_margin": tie_margin,
        "seeds": derived["evaluation"]["seeds"],
    })
    if metrics_path.is_file():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("artifact_signature") == signature and predictions_path.is_file():
            return existing

    frames = {}
    for split in ("train", "val", "test"):
        frames[split] = merge_feature_frames(base[split], cot[split])
    base_columns = base["columns"]
    _, y_train = split_xy(frames["train"], base_columns)
    _, y_val = split_xy(frames["val"], base_columns)
    existing_ml = json.loads((cell / "ml_metrics.json").read_text(encoding="utf-8"))
    parameter_source = mixture if mixture != "all" else "standard"
    base_params = existing_ml[parameter_source]["xgboost"]["params"]

    sweep = []
    for count in available_k:
        columns = [*base_columns, *ordered[:count]]
        x_train, _ = split_xy(frames["train"], columns)
        x_val, _ = split_xy(frames["val"], columns)
        model = xgb(derived, base_params, 17)
        model.fit(x_train, y_train)
        metrics = evaluate(model, x_val, y_val)
        sweep.append({
            "n_cluster_features": count,
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_accuracy": metrics["accuracy"],
            "validation_roc_auc": metrics.get("roc_auc"),
        })
    selected = choose_smallest_near_best(sweep, tie_margin)
    selected_k = int(selected["n_cluster_features"])
    columns = [*base_columns, *ordered[:selected_k]]

    if selected_k == 0:
        params = base_params
    else:
        x_train, _ = split_xy(frames["train"], columns)
        x_val, _ = split_xy(frames["val"], columns)
        params = tune_xgboost(
            x_train,
            y_train,
            x_val,
            y_val,
            derived,
            int(derived["optuna"]["n_trials"]),
            seed=17,
        )

    runs = {}
    prediction_rows: list[dict] = []
    for seed in derived["evaluation"]["seeds"]:
        model = xgb(derived, params, int(seed))
        x_train, y_train = split_xy(frames["train"], columns)
        model.fit(x_train, y_train)
        runs[str(seed)] = {}
        for split in ("train", "val", "test"):
            values, truth = split_xy(frames[split], columns)
            metrics = evaluate(model, values, truth)
            predicted = model.predict(values)
            if split == "test":
                metrics["balanced_accuracy_ci"] = _balanced_accuracy_ci(
                    truth, predicted, samples=1000, seed=int(seed)
                )
            runs[str(seed)][split] = metrics
            probabilities = model.predict_proba(values)
            for index, customer_id in enumerate(frames[split]["customer_id"]):
                row = {
                    "feature_set": feature_set,
                    "classifier": "xgboost",
                    "seed": int(seed),
                    "split": split,
                    "customer_id": int(customer_id),
                    "label": int(truth[index]),
                    "prediction": int(predicted[index]),
                }
                row.update({
                    f"probability_{int(class_id)}": float(
                        probabilities[index, probability_index]
                    )
                    for probability_index, class_id in enumerate(model.classes_)
                })
                prediction_rows.append(row)

    payload = {
        "artifact_signature": signature,
        "feature_set": feature_set,
        "base_feature_set": mixture,
        "selection": {
            "metric": "validation_balanced_accuracy",
            "tie_margin": tie_margin,
            "selected_k": selected_k,
            "selected_cluster_features": ordered[:selected_k],
            "sweep": sweep,
        },
        "xgboost": {
            "params": params,
            "runs": runs,
            "summary": {
                split: _seed_metric_summary(runs, split)
                for split in ("val", "test")
            },
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_jsonl(predictions_path, prediction_rows)
    atomic_write_json(metrics_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("results/v2/derived/reviewer-v4-offline-v2"),
    )
    parser.add_argument("--tie-margin", type=float, default=0.005)
    parser.add_argument("--candidates", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--mixtures",
        nargs="+",
        choices=tuple(MIXTURES),
        default=["handcrafted"],
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if 0 not in args.candidates or any(value < 0 for value in args.candidates):
        raise ValueError("--candidates must be non-negative and include zero")
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "wait": args.wait,
        "tie_margin": args.tie_margin,
        "candidates": sorted(set(args.candidates)),
        "mixtures": args.mixtures,
        "cells": [
            str(args.derived_root / dataset / model / "seed_17")
            for dataset, model in CELLS
        ],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    while not prerequisites_complete(args.derived_root):
        if not args.wait:
            raise RuntimeError("ML and cluster-seed prerequisites are incomplete")
        print("Waiting for ML and cluster-seed queues...", flush=True)
        time.sleep(args.poll_seconds)
    state_path = args.derived_root / "selected_concat_queue.json"
    state = {**plan, "state": "running", "completed": [], "failed": None}
    atomic_write_json(state_path, state)
    for mixture in args.mixtures:
        for dataset, model in CELLS:
            key = f"{mixture}:{dataset}:{model}"
            state["current"] = key
            atomic_write_json(state_path, state)
            try:
                run_cell(
                    args.derived_root / dataset / model / "seed_17",
                    mixture=mixture,
                    tie_margin=args.tie_margin,
                    candidates=tuple(sorted(set(args.candidates))),
                )
            except Exception as error:
                state["state"] = "failed"
                state["failed"] = {
                    "cell": key,
                    "type": type(error).__name__,
                    "message": str(error),
                }
                atomic_write_json(state_path, state)
                raise
            state["completed"].append(key)
            atomic_write_json(state_path, state)
    state["state"] = "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()

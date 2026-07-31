#!/usr/bin/env python3
"""Measure fold-pure clustering stability for Berka and Data Fusion."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from xgboost import XGBClassifier

from scripts.run_cv_offline_pipeline import _rank_mi
from src.data.benchmark_registry import BENCHMARKS
from src.data.entity_ids import canonical_entity_id
from src.experiments.artifacts import atomic_write_json
from src.experiments.config_builder import load_yaml
from src.models.ml_baseline import evaluate, primary_score, xgb_objective
from src.pipeline.cot_features import load_claim_records
from src.pipeline.semantic_features import (
    build_semantic_model_from_partition,
    fit_text_embedding_space,
    transform_precomputed_claim_space,
    transform_text_embedding_space,
    unique_claim_space,
)


def subset(records: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    rows = [
        row for row in records
        if canonical_entity_id(row["customer_id"]) in ids
    ]
    if {canonical_entity_id(row["customer_id"]) for row in rows} != ids:
        raise ValueError("Stability subset does not cover exact fold IDs")
    return rows


def assignment_agreement(reference, candidate) -> dict[str, float]:
    merged = reference[["claim_id", "cluster_index"]].merge(
        candidate[["claim_id", "cluster_index"]], on="claim_id",
        suffixes=("_reference", "_candidate"), validate="one_to_one",
    )
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("Stability variants use different claim anchors")
    left = merged["cluster_index_reference"].to_numpy(np.int32)
    right = merged["cluster_index_candidate"].to_numpy(np.int32)
    joint = (left >= 0) & (right >= 0)
    return {
        "ari": float(adjusted_rand_score(left[joint], right[joint])),
        "nmi": float(normalized_mutual_info_score(left[joint], right[joint])),
        "joint_assignment_coverage": float(joint.mean()),
    }


def fit_metrics(train, validation, test, columns, config, seed: int) -> dict:
    params = dict(
        objective=xgb_objective(config), n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        random_state=seed, n_jobs=2, tree_method="hist", verbosity=0,
        eval_metric="logloss",
    )
    inner = XGBClassifier(**params)
    inner.fit(train[columns].to_numpy(np.float32), train["label"].to_numpy(int))
    val_metrics = evaluate(
        inner, validation[columns].to_numpy(np.float32),
        validation["label"].to_numpy(int),
    )
    final = XGBClassifier(**params)
    outer = __import__("pandas").concat([train, validation], ignore_index=True)
    final.fit(outer[columns].to_numpy(np.float32), outer["label"].to_numpy(int))
    test_metrics = evaluate(
        final, test[columns].to_numpy(np.float32), test["label"].to_numpy(int),
    )
    return {
        "validation": val_metrics, "test": test_metrics,
        "validation_primary_score": primary_score(val_metrics, config),
        "test_primary_score": primary_score(test_metrics, config),
    }


def run_cell(
    *, dataset: str, model: str, fold: int, run_id: str,
    prepared_root: Path, derived_root: Path, seeds: list[int],
    cluster_counts: list[int], embedding_model: str,
) -> dict[str, Any]:
    spec = BENCHMARKS[dataset]
    generated = Path("logs/runs") / run_id / "generated" / dataset / f"fold_{fold}"
    selection_payload = json.loads(
        (generated / "prompt_selection.json").read_text(encoding="utf-8")
    )
    config = load_yaml(selection_payload["selected_configs"][model]["path"])
    source_root = Path(config["output"]["base_dir"])
    cell = derived_root / dataset / spec.protocol / f"fold_{fold}" / model
    selection = json.loads((cell / "cluster_selection.json").read_text(encoding="utf-8"))
    selected = selection["selected"]
    fold_manifest = json.loads((
        prepared_root / dataset / spec.protocol / f"fold_{fold}" / "fold_manifest.json"
    ).read_text(encoding="utf-8"))
    inner_train_ids = {
        canonical_entity_id(value) for value in fold_manifest["ids"]["inner_train"]
    }
    inner_val_ids = {
        canonical_entity_id(value) for value in fold_manifest["ids"]["inner_validation"]
    }
    outer_train_records = load_claim_records(source_root / "claims_train.jsonl")
    test_records = load_claim_records(source_root / "claims_test.jsonl")
    outer_space = unique_claim_space(outer_train_records)
    test_space = unique_claim_space(test_records)
    outer_embeddings, transformer, embedding_signature = fit_text_embedding_space(
        outer_space["texts"], embedding_model
    )
    test_embeddings = transform_text_embedding_space(
        test_space["texts"], embedding_model, transformer
    )
    outer_embeddings = np.asarray(outer_embeddings, np.float32)
    test_embeddings = np.asarray(test_embeddings, np.float32)
    selected_count = (
        int(selected["candidate"].removeprefix("k_"))
        if selected["candidate"].startswith("k_") else int(selected["n_clusters"])
    )
    variants = [(selected_count, seed, "clustering_seed") for seed in seeds]
    variants += [(count, 17, "cluster_granularity") for count in cluster_counts]
    variants = list(dict.fromkeys(variants))
    reference_assignments = None
    rows = []
    for count, seed, axis in variants:
        labels = MiniBatchKMeans(
            n_clusters=min(count, len(outer_embeddings)), batch_size=4096,
            n_init=3, max_iter=200, random_state=seed,
            reassignment_ratio=0.01,
        ).fit_predict(outer_embeddings).astype(np.int32)
        variant_config = copy.deepcopy(config)
        variant_config.setdefault("clustering", {}).update({
            "mode": "label_agnostic", "feature_encoding": selected["encoding"],
            "min_client_coverage": 5, "max_assign_distance": 0.45,
            "n_clusters": count, "embedding_model": embedding_model,
        })
        cluster_model = build_semantic_model_from_partition(
            variant_config, occurrences=outer_space["occurrences"],
            texts=outer_space["texts"],
            occurrence_to_unique=outer_space["occurrence_to_unique"],
            embeddings=outer_embeddings, raw_ids=labels,
            embedding_state_signature=embedding_signature, train_records=None,
        )
        outer_features, outer_assignments = transform_precomputed_claim_space(
            outer_train_records, outer_space, outer_embeddings, cluster_model
        )
        test_features, _ = transform_precomputed_claim_space(
            test_records, test_space, test_embeddings, cluster_model
        )
        columns = list(cluster_model["feature_names"])
        train_mask = outer_features["customer_id"].map(canonical_entity_id).isin(inner_train_ids)
        val_mask = outer_features["customer_id"].map(canonical_entity_id).isin(inner_val_ids)
        train = outer_features[train_mask].copy()
        validation = outer_features[val_mask].copy()
        ranking = _rank_mi(train, columns, selected["encoding"])
        names = ranking[: min(int(selected["n_features"]), len(ranking))]
        if count == selected_count and seed == 17:
            reference_assignments = outer_assignments
        if reference_assignments is None:
            agreement = None
        else:
            agreement = assignment_agreement(reference_assignments, outer_assignments)
        rows.append({
            "axis": axis, "cluster_count_requested": count, "seed": seed,
            "n_clusters_after_coverage": len(columns), "n_features": len(names),
            "encoding": selected["encoding"],
            "agreement_vs_selected_seed17": agreement,
            **fit_metrics(train, validation, test_features, names, config, seed),
        })
    # Reference is always first because seeds default to 17,101,947.
    if reference_assignments is None:
        raise RuntimeError("Stability queue must include reference seed 17")
    output = cell / "stability" / "final_e5"
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset, "model": model, "fold": fold,
        "protocol": spec.protocol, "embedding_model": embedding_model,
        "selected_candidate": selected["candidate"],
        "selection_split": "inner_train_to_inner_validation",
        "formation_population": "outer_train_only", "rows": rows,
    }
    atomic_write_json(output / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--models", default="qwen,gpt_oss")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("data/benchmarks_v5"))
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--seeds", default="17,101,947")
    parser.add_argument("--cluster-counts", default="100,200,400,800")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    models = [value for value in args.models.split(",") if value]
    folds = [int(value) for value in args.folds.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    counts = [int(value) for value in args.cluster_counts.split(",") if value]
    if 17 not in seeds:
        raise ValueError("Reference seed 17 is required")
    plan = {
        "mode": "execute" if args.execute else "dry-run", "dataset": args.dataset,
        "models": models, "folds": folds, "seeds": seeds,
        "cluster_counts": counts, "embedding_model": args.embedding_model,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    results = [
        run_cell(
            dataset=args.dataset, model=model, fold=fold, run_id=args.run_id,
            prepared_root=args.prepared_root, derived_root=args.derived_root,
            seeds=seeds, cluster_counts=counts,
            embedding_model=args.embedding_model,
        )
        for model in models for fold in folds
    ]
    atomic_write_json(
        args.derived_root / args.dataset / BENCHMARKS[args.dataset].protocol
        / "stability_summary.json",
        {"protocol": plan, "cells": results},
    )


if __name__ == "__main__":
    main()

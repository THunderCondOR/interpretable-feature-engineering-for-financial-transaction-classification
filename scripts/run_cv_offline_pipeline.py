#!/usr/bin/env python3
"""Fit fold-pure claim clusters and seeded ML models for v5 benchmarks."""

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
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

from src.data.benchmark_registry import BENCHMARKS
from src.data.entity_ids import canonical_entity_id
from src.experiments.artifacts import atomic_write_json, fingerprint
from src.experiments.config_builder import load_yaml
from src.models.ml_baseline import (
    evaluate,
    primary_score,
    run_ml_baseline,
    xgb_objective,
)
from src.models.optional_boosters import run_optional_boosters
from src.pipeline.cot_features import load_claim_records
from src.pipeline.semantic_features import (
    build_semantic_model_from_partition,
    cut_agglomerative_hierarchy,
    fit_agglomerative_hierarchy,
    fit_text_embedding_space,
    transform_precomputed_claim_space,
    transform_text_embedding_space,
    unique_claim_space,
)


CANDIDATES = ("threshold_0.01", "k_100", "k_200", "k_400", "k_800")
ENCODINGS = ("binary", "normalized_count", "raw_count")
TOP_K = (50, 100, 150, 200)


def _subset(
    records: list[dict[str, Any]], ids: set[int | str]
) -> list[dict[str, Any]]:
    rows = [
        row for row in records
        if canonical_entity_id(row["customer_id"]) in ids
    ]
    if {canonical_entity_id(row["customer_id"]) for row in rows} != ids:
        raise ValueError("Claims do not exactly cover the requested fold subset")
    return rows


def _partition(
    candidate: str,
    embeddings: np.ndarray,
    *,
    hierarchy: dict[str, Any] | None,
) -> np.ndarray:
    if candidate.startswith("k_"):
        count = min(int(candidate.removeprefix("k_")), len(embeddings))
        return MiniBatchKMeans(
            n_clusters=count,
            batch_size=4096,
            n_init=3,
            max_iter=200,
            random_state=17,
            reassignment_ratio=0.01,
        ).fit_predict(np.asarray(embeddings, dtype=np.float32))
    if hierarchy is None:
        raise ValueError("Threshold candidate requires an agglomerative hierarchy")
    return cut_agglomerative_hierarchy(
        hierarchy,
        distance_threshold=float(candidate.removeprefix("threshold_")),
    )


def _model_config(config: dict[str, Any], candidate: str, encoding: str) -> dict:
    result = copy.deepcopy(config)
    result.setdefault("clustering", {}).update({
        "mode": "label_agnostic",
        "feature_encoding": encoding,
        "min_client_coverage": 5,
        "max_assign_distance": 0.45,
        "distance_threshold": 0.01,
        "n_clusters": (
            int(candidate.removeprefix("k_"))
            if candidate.startswith("k_") else None
        ),
    })
    result.setdefault("feature_selection", {})["enabled"] = False
    return result


def _score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    config: dict[str, Any],
) -> float:
    model = XGBClassifier(
        objective=xgb_objective(config),
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=17,
        n_jobs=2,
        tree_method="hist",
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(
        train[columns].to_numpy(np.float32),
        train["label"].to_numpy(int),
    )
    metrics = evaluate(
        model,
        validation[columns].to_numpy(np.float32),
        validation["label"].to_numpy(int),
    )
    return primary_score(metrics, config)


def _rank_mi(frame: pd.DataFrame, columns: list[str], encoding: str) -> list[str]:
    scores = mutual_info_classif(
        frame[columns].to_numpy(np.float32),
        frame["label"].to_numpy(int),
        discrete_features=encoding != "normalized_count",
        random_state=17,
    )
    return [
        name for name, _ in sorted(
            zip(columns, scores), key=lambda item: (-float(item[1]), item[0])
        )
    ]


def _fit_spaces(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict, dict, np.ndarray, np.ndarray, str]:
    train_space = unique_claim_space(train_records)
    validation_space = unique_claim_space(validation_records)
    embedding_model = config.get("clustering", {}).get(
        "embedding_model",
        config.get("pipeline", {}).get("embedding_model", "all-MiniLM-L6-v2"),
    )
    train_embeddings, transformer, signature = fit_text_embedding_space(
        train_space["texts"], embedding_model
    )
    validation_embeddings = transform_text_embedding_space(
        validation_space["texts"], embedding_model, transformer
    )
    return (
        train_space,
        validation_space,
        np.asarray(train_embeddings, dtype=np.float32),
        np.asarray(validation_embeddings, dtype=np.float32),
        signature,
    )


def select_representation(
    *,
    config: dict[str, Any],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    exact_claim_limit: int,
) -> dict[str, Any]:
    (
        train_space, validation_space, train_embeddings,
        validation_embeddings, embedding_signature,
    ) = _fit_spaces(train_records, validation_records, config)
    hierarchy = (
        fit_agglomerative_hierarchy(train_embeddings)
        if len(train_embeddings) <= exact_claim_limit else None
    )
    candidates = [
        candidate for candidate in CANDIDATES
        if hierarchy is not None or candidate.startswith("k_")
    ]
    rows = []
    for candidate in candidates:
        raw_ids = _partition(candidate, train_embeddings, hierarchy=hierarchy)
        binary_config = _model_config(config, candidate, "binary")
        try:
            model = build_semantic_model_from_partition(
                binary_config,
                occurrences=train_space["occurrences"],
                texts=train_space["texts"],
                occurrence_to_unique=train_space["occurrence_to_unique"],
                embeddings=train_embeddings,
                raw_ids=raw_ids,
                embedding_state_signature=embedding_signature,
                train_records=None,
            )
        except ValueError as exc:
            rows.append({
                "candidate": candidate,
                "status": "invalid",
                "reason": str(exc),
            })
            continue
        for encoding in ENCODINGS:
            model["settings"]["feature_encoding"] = encoding
            train_features, _ = transform_precomputed_claim_space(
                train_records, train_space, train_embeddings, model
            )
            validation_features, _ = transform_precomputed_claim_space(
                validation_records,
                validation_space,
                validation_embeddings,
                model,
            )
            columns = list(model["feature_names"])
            ranking = _rank_mi(train_features, columns, encoding)
            sizes = [value for value in TOP_K if value < len(columns)] + [
                len(columns)
            ]
            for size in sorted(set(sizes)):
                selected = ranking[:size]
                rows.append({
                    "candidate": candidate,
                    "encoding": encoding,
                    "n_clusters": len(columns),
                    "n_features": len(selected),
                    "validation_score": _score(
                        train_features,
                        validation_features,
                        selected,
                        config,
                    ),
                })
    valid_rows = [row for row in rows if row.get("status") != "invalid"]
    if not valid_rows:
        raise RuntimeError("No clustering candidate produced usable features")
    best = max(row["validation_score"] for row in valid_rows)
    eligible = [
        row for row in valid_rows
        if best - row["validation_score"] <= 0.005
    ]
    selected = sorted(
        eligible,
        key=lambda row: (
            row["n_features"], row["n_clusters"],
            {"binary": 0, "normalized_count": 1, "raw_count": 2}[
                row["encoding"]
            ],
            -row["validation_score"],
        ),
    )[0]
    return {
        "primary_metric": config["dataset"]["primary_metric"],
        "tie_margin": 0.005,
        "candidate_results": rows,
        "selected": selected,
    }


def fit_final_space(
    *,
    config: dict[str, Any],
    outer_train_records: list[dict[str, Any]],
    outer_test_records: list[dict[str, Any]],
    inner_train_ids: set[int | str],
    inner_validation_ids: set[int | str],
    selection: dict[str, Any],
    output_root: Path,
    exact_claim_limit: int,
) -> None:
    (
        train_space, test_space, train_embeddings, test_embeddings,
        embedding_signature,
    ) = _fit_spaces(outer_train_records, outer_test_records, config)
    chosen = selection["selected"]
    hierarchy = (
        fit_agglomerative_hierarchy(train_embeddings)
        if chosen["candidate"].startswith("threshold_")
        and len(train_embeddings) <= exact_claim_limit
        else None
    )
    raw_ids = _partition(
        chosen["candidate"], train_embeddings, hierarchy=hierarchy
    )
    final_config = _model_config(
        config, chosen["candidate"], chosen["encoding"]
    )
    model = build_semantic_model_from_partition(
        final_config,
        occurrences=train_space["occurrences"],
        texts=train_space["texts"],
        occurrence_to_unique=train_space["occurrence_to_unique"],
        embeddings=train_embeddings,
        raw_ids=raw_ids,
        embedding_state_signature=embedding_signature,
        train_records=None,
    )
    outer_train_features, train_assignments = transform_precomputed_claim_space(
        outer_train_records, train_space, train_embeddings, model
    )
    test_features, test_assignments = transform_precomputed_claim_space(
        outer_test_records, test_space, test_embeddings, model
    )
    cluster_columns = list(model["feature_names"])
    inner_train_mask = outer_train_features["customer_id"].map(
        canonical_entity_id
    ).isin(inner_train_ids)
    inner_validation_mask = outer_train_features["customer_id"].map(
        canonical_entity_id
    ).isin(inner_validation_ids)
    inner_train = outer_train_features[inner_train_mask].copy()
    inner_validation = outer_train_features[inner_validation_mask].copy()
    ranking = _rank_mi(inner_train, cluster_columns, chosen["encoding"])
    selected_names = ranking[: min(chosen["n_features"], len(ranking))]
    model["selected_feature_names"] = selected_names
    model["selection_mode"] = "inner_train_mutual_information"
    model["representation"] = chosen
    output_root.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value for key, value in model.items()
        if key not in {"centroids", "embedding_transformer"}
    }
    atomic_write_json(output_root / "cot_clusters.json", serializable)
    centroids_path = output_root / "cot_cluster_model.npz"
    temporary_centroids = centroids_path.with_suffix(".npz.tmp")
    with temporary_centroids.open("wb") as file:
        np.savez_compressed(file, centroids=model["centroids"])
    temporary_centroids.replace(centroids_path)
    for split, frame in (
        ("train", inner_train),
        ("val", inner_validation),
        ("test", test_features),
        ("outer_train", outer_train_features),
    ):
        path = output_root / f"cot_features_{split}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        frame[["customer_id", "label", *selected_names]].to_parquet(
            temporary, index=False
        )
        temporary.replace(path)
    for name, assignments in (
        ("outer_train", train_assignments), ("test", test_assignments)
    ):
        path = output_root / f"claim_assignments_{name}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        assignments.to_parquet(temporary, index=False)
        temporary.replace(path)
    atomic_write_json(output_root / "cluster_selection.json", selection)
    atomic_write_json(output_root / "cluster_completion.json", {
        "status": "completed",
        "formation_population": "outer_train_only",
        "selection_population": "inner_train_to_inner_validation",
        "test_transform": "frozen_outer_train_centroids",
        "n_clusters": len(model["feature_names"]),
        "n_selected_features": len(selected_names),
        "train_assignment_coverage": float(train_assignments["assigned"].mean()),
        "test_assignment_coverage": float(test_assignments["assigned"].mean()),
        "signature": fingerprint({
            "selection": selection,
            "formation_signature": model["formation_signature"],
            "selected_features": selected_names,
        }),
    })


def run_fold(
    *,
    dataset: str,
    model: str,
    fold: int,
    run_id: str,
    prepared_root: Path,
    derived_root: Path,
    exact_claim_limit: int,
    skip_ml: bool,
) -> None:
    spec = BENCHMARKS[dataset]
    generated = (
        Path("logs/runs") / run_id / "generated" / dataset / f"fold_{fold}"
    )
    selection_path = generated / "prompt_selection.json"
    prompt_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_config = load_yaml(
        prompt_selection["selected_configs"][model]["path"]
    )
    source_root = Path(selected_config["output"]["base_dir"])
    fold_manifest = json.loads(
        (
            prepared_root / dataset / spec.protocol / f"fold_{fold}"
            / "fold_manifest.json"
        ).read_text(encoding="utf-8")
    )
    outer_train = load_claim_records(source_root / "claims_train.jsonl")
    outer_test = load_claim_records(source_root / "claims_test.jsonl")
    inner_train_ids = {
        canonical_entity_id(value)
        for value in fold_manifest["ids"]["inner_train"]
    }
    inner_validation_ids = {
        canonical_entity_id(value)
        for value in fold_manifest["ids"]["inner_validation"]
    }
    tuning = select_representation(
        config=selected_config,
        train_records=_subset(outer_train, inner_train_ids),
        validation_records=_subset(outer_train, inner_validation_ids),
        exact_claim_limit=exact_claim_limit,
    )
    output_root = (
        derived_root / dataset / spec.protocol / f"fold_{fold}" / model
    )
    fit_final_space(
        config=selected_config,
        outer_train_records=outer_train,
        outer_test_records=outer_test,
        inner_train_ids=inner_train_ids,
        inner_validation_ids=inner_validation_ids,
        selection=tuning,
        output_root=output_root,
        exact_claim_limit=exact_claim_limit,
    )
    if skip_ml:
        return
    ml_config = copy.deepcopy(selected_config)
    events = str(
        prepared_root / dataset / spec.protocol / "events.parquet"
    )
    fold_root = (
        prepared_root / dataset / spec.protocol / f"fold_{fold}"
    )
    ml_config["dataset"]["splits"] = {
        "train": events, "val": events, "test": events,
    }
    ml_config["dataset"]["client_ids_by_split"] = {
        "train": str(fold_root / "inner_train_ids.json"),
        "val": str(fold_root / "inner_validation_ids.json"),
        "test": str(fold_root / "outer_test_ids.json"),
    }
    ml_config["output"]["base_dir"] = str(output_root)
    ml_config.setdefault("input", {})["cot_features_base_dir"] = str(output_root)
    ml_config.setdefault("evaluation", {}).update({
        "seeds": [17, 101, 947],
        "refit_train_and_val_for_test": True,
        "bootstrap_samples": 1000,
    })
    ml_config.setdefault("optuna", {})["n_trials"] = 30
    run_ml_baseline(
        ml_config,
        experiments=[
            "standard", "llm_profile", "standard_profile", "handcrafted",
            "cot", "concat", "standard_cot", "all_features",
        ],
    )
    run_optional_boosters(
        ml_config,
        experiments=[
            "standard", "llm_profile", "standard_profile", "handcrafted",
            "cot", "concat", "standard_cot", "all_features",
        ],
        output_path=output_root / "optional_booster_metrics.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="berka,datafusion_education",
    )
    parser.add_argument("--models", default="qwen,gpt_oss")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--run-id", default="reviewer-v5-benchmarks")
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("data/benchmarks_v5")
    )
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("results/v5/derived/cv_main"),
    )
    parser.add_argument("--exact-claim-limit", type=int, default=45_000)
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    datasets = [value for value in args.datasets.split(",") if value]
    models = [value for value in args.models.split(",") if value]
    folds = [int(value) for value in args.folds.split(",") if value]
    unknown = set(datasets) - set(BENCHMARKS)
    if unknown or set(models) - {"qwen", "gpt_oss"}:
        raise ValueError(f"Unsupported datasets/models: {unknown}, {models}")
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "datasets": datasets,
        "models": models,
        "folds": folds,
        "selection": "inner_train -> inner_validation",
        "final_fit": "outer_train -> frozen outer_test transform",
        "ml": not args.skip_ml,
    }, indent=2))
    if not args.execute:
        return
    for dataset in datasets:
        for model in models:
            for fold in folds:
                run_fold(
                    dataset=dataset,
                    model=model,
                    fold=fold,
                    run_id=args.run_id,
                    prepared_root=args.prepared_root,
                    derived_root=args.derived_root,
                    exact_claim_limit=args.exact_claim_limit,
                    skip_ml=args.skip_ml,
                )


if __name__ == "__main__":
    main()

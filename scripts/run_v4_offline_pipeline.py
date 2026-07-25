#!/usr/bin/env python3
"""Build v4 semantic features and ML results from immutable API artifacts."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

from src.experiments.artifacts import (
    atomic_write_json,
    file_sha256,
    files_fingerprint,
    fingerprint,
)
from src.experiments.derived_artifacts import (
    compatible_stage,
    complete_stage,
    source_contract,
    stage_identity,
    verify_source_unchanged,
)
from src.experiments.events import append_event
from src.models.ml_baseline import run_ml_baseline, xgb_objective
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

DEFAULT_CANDIDATES = ("threshold_0.01", "k_200", "k_400", "k_800")
DEFAULT_EXACT_CLAIM_LIMIT = 45_000
DEFAULT_ENCODINGS = ("binary", "raw_count", "normalized_count")
DEFAULT_MI_TOP_K = (50, 100, 150, 200)


def choose_clustering_backend(
    n_claims: int,
    requested: str,
    *,
    exact_claim_limit: int = DEFAULT_EXACT_CLAIM_LIMIT,
) -> str:
    if requested not in {"auto", "agglomerative", "minibatch_kmeans"}:
        raise ValueError(f"Unsupported clustering backend: {requested}")
    if requested != "auto":
        return requested
    return (
        "agglomerative"
        if int(n_claims) <= int(exact_claim_limit)
        else "minibatch_kmeans"
    )


def compatible_candidates(
    candidates: list[str],
    backend: str,
) -> list[str]:
    if backend == "agglomerative":
        return candidates
    supported = [item for item in candidates if item.startswith("k_")]
    if not supported:
        raise ValueError(
            "minibatch_kmeans requires at least one fixed-K candidate"
        )
    return supported


def offline_event(
    output_root: Path,
    source: dict[str, Any],
    *,
    stage: str,
    state: str = "completed",
    **payload: Any,
) -> None:
    append_event(
        output_root / "events.jsonl",
        event=f"offline_{stage}",
        state=state,
        run_id="reviewer-v4-offline-v1",
        dataset=source["dataset"],
        model=source["model_slug"],
        stage=stage,
        **payload,
    )


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as file:
        np.save(file, np.asarray(values))
    temporary.replace(path)


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as file:
        np.savez_compressed(file, **values)
    temporary.replace(path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    with open(temporary, "rb") as file:
        os.fsync(file.fileno())
    temporary.replace(target)


def load_source(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    config = copy.deepcopy(manifest["config"])
    return manifest, config


def validate_claim_records(records: list[dict[str, Any]], split: str) -> None:
    identifiers = [int(row["customer_id"]) for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate clients in claims_{split}.jsonl")
    for row in records:
        if row.get("error") or row.get("error_type"):
            raise ValueError(f"Failed claim record in {split}: {row['customer_id']}")
        if not row.get("claim_records"):
            raise ValueError(f"Missing atomic claim records in {split}: {row['customer_id']}")


def serialize_space(space: dict[str, Any], path: Path) -> None:
    occurrences = space["occurrences"]
    mapping = np.asarray(space["occurrence_to_unique"], dtype=np.int32)
    occurrence_counts = np.bincount(mapping, minlength=len(space["texts"]))
    representative = {}
    for row, unique_index in zip(occurrences, mapping):
        representative.setdefault(int(unique_index), row["original_text"])
    frame = pd.DataFrame({
        "unique_index": np.arange(len(space["texts"]), dtype=np.int32),
        "normalized_text": space["texts"],
        "representative_text": [
            representative[index] for index in range(len(space["texts"]))
        ],
        "occurrences": occurrence_counts.astype(np.int32),
    })
    atomic_frame(path, frame)


def model_payload(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model.items()
        if key not in {"centroids", "embedding_transformer"}
    }


def load_model(meta_path: Path, centroids_path: Path) -> dict[str, Any]:
    model = json.loads(meta_path.read_text(encoding="utf-8"))
    model["centroids"] = np.load(centroids_path)["centroids"]
    model["embedding_transformer"] = None
    return model


def candidate_partition(
    hierarchy: dict[str, Any],
    candidate: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate.startswith("threshold_"):
        threshold = float(candidate.removeprefix("threshold_"))
        labels = cut_agglomerative_hierarchy(
            hierarchy,
            distance_threshold=threshold,
        )
        return labels, {"distance_threshold": threshold, "n_clusters": None}
    if candidate.startswith("k_"):
        count = int(candidate.removeprefix("k_"))
        labels = cut_agglomerative_hierarchy(hierarchy, n_clusters=count)
        return labels, {"distance_threshold": 0.01, "n_clusters": count}
    raise ValueError(f"Unsupported cluster candidate: {candidate}")


def fixed_candidate_score(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    columns = [column for column in train if column.startswith("cot_")]
    model = XGBClassifier(
        objective=xgb_objective(config),
        n_estimators=200,
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
    model.fit(train[columns].to_numpy(np.float32), train["label"].to_numpy(int))
    prediction = model.predict(val[columns].to_numpy(np.float32))
    return float(balanced_accuracy_score(val["label"], prediction))


def features_from_assignments(
    records: list[dict[str, Any]],
    assignments: pd.DataFrame,
    model: dict[str, Any],
    *,
    encoding: str,
    selected_feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build a client matrix from frozen claim-to-cluster assignments."""
    names = list(model["feature_names"])
    counts = (
        assignments.loc[assignments["assigned"]]
        .groupby(["customer_id", "cluster_index"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(len(names)), fill_value=0)
    )
    client_ids = [int(row["customer_id"]) for row in records]
    values = counts.reindex(client_ids, fill_value=0).to_numpy(np.float32)
    if encoding == "binary":
        values = (values > 0).astype(np.float32)
    elif encoding == "normalized_count":
        values = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    elif encoding != "raw_count":
        raise ValueError(f"Unknown feature encoding: {encoding}")
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, "label", [int(row.get("label", -1)) for row in records])
    frame.insert(0, "customer_id", client_ids)
    if selected_feature_names is not None:
        frame = frame[
            ["customer_id", "label", *selected_feature_names]
        ]
    return frame


def rank_features_by_train_mi(
    train: pd.DataFrame,
    *,
    encoding: str,
) -> list[str]:
    names = [column for column in train if column.startswith("cot_")]
    scores = mutual_info_classif(
        train[names].to_numpy(np.float32),
        train["label"].to_numpy(int),
        discrete_features=encoding != "normalized_count",
        random_state=17,
    )
    return [
        name
        for name, _ in sorted(
            zip(names, scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]


def choose_representation(
    rows: list[dict[str, Any]],
    tie_margin: float,
) -> dict[str, Any]:
    best = max(row["validation_balanced_accuracy"] for row in rows)
    eligible = [
        row
        for row in rows
        if best - row["validation_balanced_accuracy"] <= tie_margin
    ]
    encoding_order = {
        "binary": 0,
        "normalized_count": 1,
        "raw_count": 2,
    }
    return sorted(
        eligible,
        key=lambda row: (
            row["n_features"],
            encoding_order[row["encoding"]],
            -row["validation_balanced_accuracy"],
        ),
    )[0]


def sweep_representations(
    *,
    records: dict[str, list[dict[str, Any]]],
    assignments: dict[str, pd.DataFrame],
    model: dict[str, Any],
    config: dict[str, Any],
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS,
    top_k_values: tuple[int, ...] = DEFAULT_MI_TOP_K,
    tie_margin: float = 0.005,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, pd.DataFrame]]:
    rows = []
    frames_by_encoding = {}
    rankings = {}
    for encoding in encodings:
        train = features_from_assignments(
            records["train"], assignments["train"], model, encoding=encoding
        )
        val = features_from_assignments(
            records["val"], assignments["val"], model, encoding=encoding
        )
        frames_by_encoding[encoding] = {"train": train, "val": val}
        rankings[encoding] = rank_features_by_train_mi(
            train,
            encoding=encoding,
        )
        sizes = [
            size for size in top_k_values if size < len(model["feature_names"])
        ] + [None]
        for top_k in sizes:
            selected_names = (
                rankings[encoding]
                if top_k is None
                else rankings[encoding][:top_k]
            )
            score = fixed_candidate_score(
                train[["customer_id", "label", *selected_names]],
                val[["customer_id", "label", *selected_names]],
                config,
            )
            rows.append({
                "encoding": encoding,
                "mi_top_k": top_k,
                "n_features": len(selected_names),
                "validation_balanced_accuracy": score,
            })
    selected = choose_representation(rows, tie_margin)
    selected_names = (
        rankings[selected["encoding"]]
        if selected["mi_top_k"] is None
        else rankings[selected["encoding"]][: selected["mi_top_k"]]
    )
    selected = {
        **selected,
        "selected_feature_names": selected_names,
        "selection_metric": "cot_validation_balanced_accuracy",
        "selection_tie_margin": tie_margin,
    }
    return selected, rows, frames_by_encoding[selected["encoding"]]


def refresh_cluster_metadata(
    model: dict[str, Any],
    assignments: pd.DataFrame,
) -> None:
    assigned = assignments[assignments["assigned"]].copy()
    for index, row in enumerate(model["cluster_meta"]):
        cell = assigned[assigned["cluster_index"] == index]
        row["assigned_occurrences"] = int(len(cell))
        row["assigned_unique_clients"] = int(cell["customer_id"].nunique())
        row["assignment_distance_mean"] = (
            float(cell["assignment_distance"].mean()) if len(cell) else None
        )
        row["assignment_distance_p95"] = (
            float(cell["assignment_distance"].quantile(0.95)) if len(cell) else None
        )
        row["assigned_label_counts"] = {
            str(label): int(count)
            for label, count in cell["label"].value_counts().sort_index().items()
        }


def stage_path(output_root: Path, stage: str) -> Path:
    return output_root / "stages" / f"{stage}.json"


def materialize_embeddings(
    *,
    output_root: Path,
    source: dict[str, Any],
    config: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], str]:
    embedding_model = config.get("clustering", {}).get(
        "embedding_model",
        config.get("pipeline", {}).get(
            "embedding_model",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )
    spaces = {split: unique_claim_space(rows) for split, rows in records.items()}
    embedding_dir = output_root / "embeddings"
    outputs = [
        embedding_dir / f"unique_claims_{split}.parquet"
        for split in ("train", "val", "test")
    ] + [
        embedding_dir / f"embeddings_{split}.npy"
        for split in ("train", "val", "test")
    ]
    identity = stage_identity(
        stage="embeddings",
        source=source,
        inputs=source["claims"],
        configuration={"embedding_model": embedding_model, "normalization": "v1"},
        repo_root=REPO_ROOT,
    )
    manifest_path = stage_path(output_root, "embeddings")
    if compatible_stage(manifest_path, identity):
        embeddings = {
            split: np.load(
                embedding_dir / f"embeddings_{split}.npy",
                mmap_mode="r",
            )
            for split in ("train", "val", "test")
        }
        signature = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )["metrics"]["embedding_state_signature"]
        return spaces, embeddings, signature

    train_embeddings, transformer, signature = fit_text_embedding_space(
        spaces["train"]["texts"],
        embedding_model,
    )
    embeddings = {"train": np.asarray(train_embeddings, dtype=np.float32)}
    for split in ("val", "test"):
        embeddings[split] = np.asarray(
            transform_text_embedding_space(
                spaces[split]["texts"],
                embedding_model,
                transformer,
            ),
            dtype=np.float32,
        )
    for split in ("train", "val", "test"):
        serialize_space(
            spaces[split],
            embedding_dir / f"unique_claims_{split}.parquet",
        )
        atomic_npy(
            embedding_dir / f"embeddings_{split}.npy",
            embeddings[split],
        )
    if transformer is not None:
        transformer_path = embedding_dir / "embedding_transformer.joblib"
        joblib.dump(transformer, transformer_path)
        outputs.append(transformer_path)
    complete_stage(
        manifest_path,
        identity,
        outputs=outputs,
        metrics={
            "embedding_model": embedding_model,
            "embedding_state_signature": signature,
            "unique_claims": {
                split: len(spaces[split]["texts"]) for split in spaces
            },
        },
    )
    return spaces, embeddings, signature


def materialize_hierarchy(
    *,
    output_root: Path,
    source: dict[str, Any],
    train_embeddings: np.ndarray,
    embedding_stage: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    hierarchy_path = output_root / "hierarchy" / "train_hierarchy.npz"
    identity = stage_identity(
        stage="hierarchy",
        source=source,
        inputs={
            "embedding_stage_signature": embedding_stage["stage_signature"],
            "train_embeddings": files_fingerprint([
                output_root / "embeddings" / "embeddings_train.npy"
            ]),
        },
        configuration={
            "backend": backend,
            "metric": "cosine" if backend == "agglomerative" else "euclidean_on_unit_vectors",
            "linkage": "average" if backend == "agglomerative" else None,
        },
        repo_root=REPO_ROOT,
    )
    manifest_path = stage_path(output_root, "hierarchy")
    if compatible_stage(manifest_path, identity):
        payload = np.load(hierarchy_path)
        return {
            "children": payload["children"],
            "distances": payload["distances"],
            "n_samples": int(payload["n_samples"][0]),
            "backend": backend,
        }
    if backend == "agglomerative":
        hierarchy = fit_agglomerative_hierarchy(train_embeddings)
    else:
        hierarchy = {
            "children": np.empty((0, 2), dtype=np.int64),
            "distances": np.empty(0, dtype=np.float64),
            "n_samples": int(len(train_embeddings)),
        }
    atomic_npz(
        hierarchy_path,
        children=hierarchy["children"],
        distances=hierarchy["distances"],
        n_samples=np.asarray([hierarchy["n_samples"]], dtype=np.int64),
    )
    complete_stage(
        manifest_path,
        identity,
        outputs=[hierarchy_path],
        metrics={
            "backend": backend,
            "n_samples": hierarchy["n_samples"],
            "estimated_condensed_distance_gb": (
                hierarchy["n_samples"] * (hierarchy["n_samples"] - 1) * 8
                / 2 / 1e9
            ) if backend == "agglomerative" else 0.0,
        },
    )
    return {**hierarchy, "backend": backend}


def run_candidate(
    *,
    candidate: str,
    output_root: Path,
    source: dict[str, Any],
    config: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    spaces: dict[str, dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    hierarchy: dict[str, Any],
    embedding_signature: str,
    hierarchy_signature: str,
) -> dict[str, Any]:
    candidate_dir = output_root / "candidates" / candidate
    if hierarchy["backend"] == "agglomerative":
        labels, overlay = candidate_partition(hierarchy, candidate)
        overlay["algorithm"] = "agglomerative_average_cosine"
    else:
        if not candidate.startswith("k_"):
            raise ValueError(
                f"{candidate} is not supported by minibatch_kmeans"
            )
        count = min(
            int(candidate.removeprefix("k_")),
            int(hierarchy["n_samples"]),
        )
        labels = MiniBatchKMeans(
            n_clusters=count,
            batch_size=4096,
            n_init=3,
            max_iter=200,
            random_state=17,
            reassignment_ratio=0.01,
        ).fit_predict(np.asarray(embeddings["train"], dtype=np.float32))
        overlay = {
            # Retained as inert compatibility metadata; fixed-K formation does
            # not consult this threshold.
            "distance_threshold": 0.01,
            "n_clusters": count,
            "algorithm": "minibatch_kmeans_unit_embeddings",
        }
    candidate_config = copy.deepcopy(config)
    candidate_config.setdefault("clustering", {}).update({
        **overlay,
        "mode": "label_agnostic",
        "feature_encoding": "binary",
        "min_client_coverage": 5,
    })
    identity = stage_identity(
        stage=f"candidate:{candidate}",
        source=source,
        inputs={
            "embedding_signature": embedding_signature,
            "hierarchy_signature": hierarchy_signature,
        },
        configuration=candidate_config["clustering"],
        repo_root=REPO_ROOT,
    )
    manifest_path = stage_path(output_root, f"candidate_{candidate}")
    meta_path = candidate_dir / "cluster_model.json"
    centroids_path = candidate_dir / "centroids.npz"
    train_path = candidate_dir / "cot_features_train.parquet"
    val_path = candidate_dir / "cot_features_val.parquet"
    train_assignments_path = candidate_dir / "claim_assignments_train.parquet"
    val_assignments_path = candidate_dir / "claim_assignments_val.parquet"
    outputs = [
        meta_path,
        centroids_path,
        train_path,
        val_path,
        train_assignments_path,
        val_assignments_path,
    ]
    if compatible_stage(manifest_path, identity):
        stage = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "candidate": candidate,
            **stage["metrics"],
            "candidate_dir": str(candidate_dir),
        }

    model = build_semantic_model_from_partition(
        candidate_config,
        occurrences=spaces["train"]["occurrences"],
        texts=spaces["train"]["texts"],
        occurrence_to_unique=spaces["train"]["occurrence_to_unique"],
        embeddings=embeddings["train"],
        raw_ids=labels,
        embedding_state_signature=embedding_signature,
        train_records=None,
    )
    train_features, train_assignments = transform_precomputed_claim_space(
        records["train"],
        spaces["train"],
        embeddings["train"],
        model,
    )
    refresh_cluster_metadata(model, train_assignments)
    val_features, val_assignments = transform_precomputed_claim_space(
        records["val"],
        spaces["val"],
        embeddings["val"],
        model,
    )
    score = fixed_candidate_score(train_features, val_features, config)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(meta_path, model_payload(model))
    atomic_npz(centroids_path, centroids=model["centroids"])
    atomic_frame(train_path, train_features)
    atomic_frame(val_path, val_features)
    atomic_frame(train_assignments_path, train_assignments)
    atomic_frame(val_assignments_path, val_assignments)
    metrics = {
        "validation_balanced_accuracy": score,
        "n_clusters": len(model["feature_names"]),
        "train_assignment_coverage": float(train_assignments["assigned"].mean()),
        "val_assignment_coverage": float(val_assignments["assigned"].mean()),
    }
    complete_stage(
        manifest_path,
        identity,
        outputs=outputs,
        metrics=metrics,
    )
    return {
        "candidate": candidate,
        **metrics,
        "candidate_dir": str(candidate_dir),
    }


def choose_candidate(rows: list[dict[str, Any]], tie_margin: float) -> dict[str, Any]:
    best_score = max(row["validation_balanced_accuracy"] for row in rows)
    eligible = [
        row
        for row in rows
        if best_score - row["validation_balanced_accuracy"] <= tie_margin
    ]
    return sorted(
        eligible,
        key=lambda row: (
            row["n_clusters"],
            -row["validation_balanced_accuracy"],
            row["candidate"],
        ),
    )[0]


def materialize_selected(
    *,
    output_root: Path,
    source: dict[str, Any],
    config: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    spaces: dict[str, dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    tie_margin: float,
) -> dict[str, Any]:
    selected_cluster = choose_candidate(rows, tie_margin)
    candidate_dir = Path(selected_cluster["candidate_dir"])
    model = load_model(
        candidate_dir / "cluster_model.json",
        candidate_dir / "centroids.npz",
    )
    train_assignments = pd.read_parquet(
        candidate_dir / "claim_assignments_train.parquet"
    )
    val_assignments = pd.read_parquet(
        candidate_dir / "claim_assignments_val.parquet"
    )
    representation, representation_rows, selected_frames = (
        sweep_representations(
            records=records,
            assignments={
                "train": train_assignments,
                "val": val_assignments,
            },
            model=model,
            config=config,
            tie_margin=tie_margin,
        )
    )
    selection_payload = {
        "selection_split": "val",
        "cluster_selection_metric": "balanced_accuracy",
        "tie_margin": tie_margin,
        "candidates": rows,
        "selected_candidate": selected_cluster["candidate"],
        "selected_validation_balanced_accuracy": selected_cluster[
            "validation_balanced_accuracy"
        ],
        "selected_n_clusters": selected_cluster["n_clusters"],
        "representation_candidates": representation_rows,
        "selected_representation": representation,
    }
    selection_payload["selection_signature"] = fingerprint(selection_payload)
    selection_path = output_root / "cluster_selection.json"
    identity = stage_identity(
        stage="selected_features",
        source=source,
        inputs={"selection_signature": selection_payload["selection_signature"]},
        configuration={
            "selected_candidate": selected_cluster["candidate"],
            "selected_representation": representation,
            "max_assign_distance": model["settings"]["max_assign_distance"],
        },
        repo_root=REPO_ROOT,
    )
    manifest_path = stage_path(output_root, "selected_features")
    if compatible_stage(manifest_path, identity):
        return selection_payload

    atomic_write_json(selection_path, selection_payload)
    test_features, test_assignments = transform_precomputed_claim_space(
        records["test"],
        spaces["test"],
        embeddings["test"],
        model,
    )
    del test_features
    selected_names = representation["selected_feature_names"]
    train_features = selected_frames["train"][
        ["customer_id", "label", *selected_names]
    ]
    val_features = selected_frames["val"][
        ["customer_id", "label", *selected_names]
    ]
    test_features = features_from_assignments(
        records["test"],
        test_assignments,
        model,
        encoding=representation["encoding"],
        selected_feature_names=selected_names,
    )
    model["selected_feature_names"] = selected_names
    model["representation"] = {
        key: value
        for key, value in representation.items()
        if key != "selected_feature_names"
    }
    selected_dir = output_root / "selected_clusters"
    selected_dir.mkdir(parents=True, exist_ok=True)
    outputs = [selection_path]
    selected_model_path = selected_dir / "cluster_model.json"
    selected_centroids_path = selected_dir / "centroids.npz"
    atomic_write_json(selected_model_path, model_payload(model))
    atomic_copy(candidate_dir / "centroids.npz", selected_centroids_path)
    outputs.extend([selected_model_path, selected_centroids_path])
    for split, features, split_assignments in (
        ("train", train_features, train_assignments),
        ("val", val_features, val_assignments),
    ):
        feature_path = output_root / f"cot_features_{split}.parquet"
        assignment_path = output_root / f"claim_assignments_{split}.parquet"
        atomic_frame(feature_path, features)
        atomic_frame(assignment_path, split_assignments)
        outputs.extend([feature_path, assignment_path])
    test_feature_path = output_root / "cot_features_test.parquet"
    test_assignment_path = output_root / "claim_assignments_test.parquet"
    atomic_frame(test_feature_path, test_features)
    atomic_frame(test_assignment_path, test_assignments)
    outputs.extend([test_feature_path, test_assignment_path])

    # Compatibility names consumed by the existing ML evaluator.
    atomic_copy(
        selected_model_path,
        output_root / "cot_clusters.json",
    )
    atomic_copy(
        selected_centroids_path,
        output_root / "cot_cluster_model.npz",
    )
    outputs.extend([
        output_root / "cot_clusters.json",
        output_root / "cot_cluster_model.npz",
    ])
    complete_stage(
        manifest_path,
        identity,
        outputs=outputs,
        metrics={
            **selection_payload,
            "test_assignment_coverage": float(
                test_assignments["assigned"].mean()
            ),
        },
    )
    return selection_payload


def run_ml(
    *,
    output_root: Path,
    source: dict[str, Any],
    config: dict[str, Any],
) -> None:
    derived = copy.deepcopy(config)
    derived["output"]["base_dir"] = str(output_root)
    derived.setdefault("input", {})["cot_features_base_dir"] = str(output_root)
    derived.setdefault("evaluation", {})["seeds"] = [17, 101, 947]
    derived["evaluation"]["bootstrap_samples"] = 1000
    derived.setdefault("optuna", {})["n_trials"] = 30
    derived.setdefault("experiment", {})["run_id"] = "reviewer-v4-offline-v1"
    identity = stage_identity(
        stage="ml",
        source=source,
        inputs=files_fingerprint([
            output_root / f"cot_features_{split}.parquet"
            for split in ("train", "val", "test")
        ]),
        configuration={
            "experiments": ["standard", "handcrafted", "cot", "concat"],
            "evaluation": derived["evaluation"],
            "optuna": derived["optuna"],
        },
        repo_root=REPO_ROOT,
    )
    manifest_path = stage_path(output_root, "ml")
    outputs = [output_root / "ml_metrics.json", output_root / "ml_predictions.jsonl"]
    if compatible_stage(manifest_path, identity):
        return
    run_ml_baseline(
        derived,
        experiments=["standard", "handcrafted", "cot", "concat"],
    )
    complete_stage(manifest_path, identity, outputs=outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("results/v2/derived/reviewer-v4-offline-v2"),
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=list(DEFAULT_CANDIDATES),
    )
    parser.add_argument("--tie-margin", type=float, default=0.005)
    parser.add_argument(
        "--clustering-backend",
        choices=("auto", "agglomerative", "minibatch_kmeans"),
        default="auto",
    )
    parser.add_argument(
        "--exact-claim-limit",
        type=int,
        default=DEFAULT_EXACT_CLAIM_LIMIT,
    )
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    source = source_contract(args.source_root)
    output_root = (
        args.derived_root
        / str(source["dataset"])
        / str(source["model_slug"])
        / f"seed_{source['seed']}"
    )
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "source": source,
        "output_root": str(output_root),
        "candidates": args.candidates,
        "clustering_backend": args.clustering_backend,
        "exact_claim_limit": args.exact_claim_limit,
        "selection": "validation balanced accuracy; simplest within 0.005",
        "ml": not args.skip_ml,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.execute:
        return

    manifest, config = load_source(args.source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "source_manifest.json", {
        "source_contract": source,
        "source_manifest": manifest,
    })
    records = {
        split: load_claim_records(args.source_root / f"claims_{split}.jsonl")
        for split in ("train", "val", "test")
    }
    for split, rows in records.items():
        validate_claim_records(rows, split)

    spaces, embeddings, embedding_signature = materialize_embeddings(
        output_root=output_root,
        source=source,
        config=config,
        records=records,
    )
    offline_event(output_root, source, stage="embeddings")
    embedding_stage = json.loads(
        stage_path(output_root, "embeddings").read_text(encoding="utf-8")
    )
    backend = choose_clustering_backend(
        len(spaces["train"]["texts"]),
        args.clustering_backend,
        exact_claim_limit=args.exact_claim_limit,
    )
    candidates = compatible_candidates(args.candidates, backend)
    print(json.dumps({
        "selected_clustering_backend": backend,
        "train_unique_claims": len(spaces["train"]["texts"]),
        "effective_candidates": candidates,
        "skipped_candidates": [
            item for item in args.candidates if item not in candidates
        ],
    }, indent=2))
    hierarchy = materialize_hierarchy(
        output_root=output_root,
        source=source,
        train_embeddings=embeddings["train"],
        embedding_stage=embedding_stage,
        backend=backend,
    )
    offline_event(output_root, source, stage="hierarchy")
    hierarchy_stage = json.loads(
        stage_path(output_root, "hierarchy").read_text(encoding="utf-8")
    )
    rows = [
        run_candidate(
            candidate=candidate,
            output_root=output_root,
            source=source,
            config=config,
            records=records,
            spaces=spaces,
            embeddings=embeddings,
            hierarchy=hierarchy,
            embedding_signature=embedding_signature,
            hierarchy_signature=hierarchy_stage["stage_signature"],
        )
        for candidate in candidates
    ]
    offline_event(
        output_root,
        source,
        stage="cluster_candidates",
        completed=len(rows),
        expected=len(candidates),
    )
    selection = materialize_selected(
        output_root=output_root,
        source=source,
        config=config,
        records=records,
        spaces=spaces,
        embeddings=embeddings,
        rows=rows,
        tie_margin=args.tie_margin,
    )
    offline_event(
        output_root,
        source,
        stage="cluster_selection",
        selected=selection["selected_candidate"],
    )
    if not args.skip_ml:
        run_ml(
            output_root=output_root,
            source=source,
            config=config,
        )
        offline_event(output_root, source, stage="ml")
    verify_source_unchanged(source)
    print(json.dumps({
        "status": "completed",
        "output_root": str(output_root),
        "selection": selection,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

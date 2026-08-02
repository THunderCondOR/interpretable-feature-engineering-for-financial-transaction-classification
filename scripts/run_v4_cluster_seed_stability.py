#!/usr/bin/env python3
"""Rerun frozen v4 clustering with additional seeds and measure stability."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scripts.run_v4_offline_pipeline import (
    atomic_frame,
    atomic_npz,
    fixed_candidate_score,
    load_source,
    model_payload,
    rank_features_by_train_mi,
    features_from_assignments,
)
from src.experiments.artifacts import atomic_write_json
from src.pipeline.cot_features import load_claim_records
from src.pipeline.semantic_features import (
    build_semantic_model_from_partition,
    cut_agglomerative_hierarchy,
    fit_agglomerative_hierarchy,
    transform_precomputed_claim_space,
    unique_claim_space,
)


CELLS = (
    ("rosbank", "qwen"),
    ("gender", "qwen"),
    ("age", "qwen"),
    ("rosbank", "gpt_oss"),
    ("gender", "gpt_oss"),
    ("age", "gpt_oss"),
)


def partition_for_seed(
    embeddings: np.ndarray,
    *,
    candidate: str,
    backend: str,
    seed: int,
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if backend == "minibatch_kmeans":
        if not candidate.startswith("k_"):
            raise ValueError(
                "MiniBatch seed stability requires a fixed-K candidate"
            )
        count = min(int(candidate.removeprefix("k_")), len(values))
        return MiniBatchKMeans(
            n_clusters=count,
            batch_size=4096,
            n_init=3,
            max_iter=200,
            random_state=seed,
            reassignment_ratio=0.01,
        ).fit_predict(values).astype(np.int32)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(values))
    hierarchy = fit_agglomerative_hierarchy(values[permutation])
    if candidate.startswith("k_"):
        labels_permuted = cut_agglomerative_hierarchy(
            hierarchy,
            n_clusters=int(candidate.removeprefix("k_")),
        )
    elif candidate.startswith("threshold_"):
        labels_permuted = cut_agglomerative_hierarchy(
            hierarchy,
            distance_threshold=float(
                candidate.removeprefix("threshold_")
            ),
        )
    else:
        raise ValueError(f"Unknown candidate: {candidate}")
    labels = np.empty(len(values), dtype=np.int32)
    labels[permutation] = labels_permuted
    return labels


def assignment_agreement(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, float]:
    columns = ["claim_id", "cluster_index", "assigned"]
    merged = reference[columns].merge(
        candidate[columns],
        on="claim_id",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("Clustering seeds use different claim anchors")
    left = merged["cluster_index_reference"].to_numpy(np.int32)
    right = merged["cluster_index_candidate"].to_numpy(np.int32)
    joint = (left >= 0) & (right >= 0)
    return {
        "ari": float(adjusted_rand_score(left[joint], right[joint])),
        "nmi": float(normalized_mutual_info_score(left[joint], right[joint])),
        "joint_assignment_coverage": float(joint.mean()),
        "candidate_assignment_coverage": float((right >= 0).mean()),
    }


def stability_output_root(cell: Path, backend_override: str | None) -> Path:
    """Keep backend-comparison artifacts separate from selected-backend results."""
    name = (
        "cluster_seeds"
        if backend_override is None
        else f"cluster_seeds_{backend_override}"
    )
    return cell / "stability" / name


def run_cell(
    cell: Path,
    seeds: list[int],
    *,
    backend_override: str | None = None,
    embedding_cache_cell: Path | None = None,
) -> dict:
    selection = json.loads(
        (cell / "cluster_selection.json").read_text(encoding="utf-8")
    )
    source_payload = json.loads(
        (cell / "source_manifest.json").read_text(encoding="utf-8")
    )
    source_root = Path(
        source_payload["source_contract"]["source_root"]
    )
    _, config = load_source(source_root)
    records = {
        split: load_claim_records(source_root / f"claims_{split}.jsonl")
        for split in ("train", "val", "test")
    }
    spaces = {
        split: unique_claim_space(records[split])
        for split in ("train", "val", "test")
    }
    embedding_cell = embedding_cache_cell or cell
    embedding_stage_path = embedding_cell / "stages" / "embeddings.json"
    if embedding_cache_cell is not None:
        cache_stage = json.loads(embedding_stage_path.read_text(encoding="utf-8"))
        if cache_stage.get("source") != source_payload["source_contract"]:
            raise ValueError(
                f"Embedding cache source mismatch: {embedding_cache_cell}"
            )
        for split in ("train", "val", "test"):
            cached_claims = pd.read_parquet(
                embedding_cell / "embeddings" / f"unique_claims_{split}.parquet",
                columns=["normalized_text"],
            )["normalized_text"].astype(str).tolist()
            if cached_claims != list(spaces[split]["texts"]):
                raise ValueError(
                    f"Embedding cache claim order mismatch for {split}: "
                    f"{embedding_cache_cell}"
                )
    embeddings = {
        split: np.load(
            embedding_cell / "embeddings" / f"embeddings_{split}.npy",
            mmap_mode="r",
        )
        for split in ("train", "val", "test")
    }
    for split, values in embeddings.items():
        if len(values) != len(spaces[split]["texts"]):
            raise ValueError(
                f"Embedding row mismatch for {split}: {len(values)} != "
                f"{len(spaces[split]['texts'])}"
            )
    selected_backend = json.loads(
        (cell / "stages" / "hierarchy.json").read_text(encoding="utf-8")
    )["metrics"]["backend"]
    backend = backend_override or selected_backend
    selected_candidate = selection["selected_candidate"]
    representation = selection["selected_representation"]
    output_root = stability_output_root(cell, backend_override)
    reference_seed = 17
    requested_seeds = list(dict.fromkeys(seeds))
    materialization_seeds = (
        [reference_seed, *[seed for seed in requested_seeds if seed != reference_seed]]
        if backend_override is not None
        else requested_seeds
    )
    reference = (
        None
        if backend_override is not None
        else pd.read_parquet(cell / "claim_assignments_train.parquet")
    )
    rows = []
    for seed in materialization_seeds:
        seed_root = output_root / f"seed_{seed}"
        summary_path = seed_root / "summary.json"
        assignments_path = seed_root / "claim_assignments_train.parquet"
        test_features_path = seed_root / "cot_features_test.parquet"
        test_assignments_path = seed_root / "claim_assignments_test.parquet"
        if all(path.is_file() for path in (
            summary_path,
            assignments_path,
            test_features_path,
            test_assignments_path,
        )):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("backend") == backend
                and summary.get("candidate") == selected_candidate
            ):
                if seed == reference_seed and reference is None:
                    reference = pd.read_parquet(assignments_path)
                rows.append(summary)
                continue
        labels = partition_for_seed(
            embeddings["train"],
            candidate=selected_candidate,
            backend=backend,
            seed=seed,
        )
        seeded_config = copy.deepcopy(config)
        seeded_config.setdefault("clustering", {}).update({
            "mode": "label_agnostic",
            "feature_encoding": representation["encoding"],
            "min_client_coverage": 5,
            "n_clusters": (
                int(selected_candidate.removeprefix("k_"))
                if selected_candidate.startswith("k_")
                else None
            ),
        })
        model = build_semantic_model_from_partition(
            seeded_config,
            occurrences=spaces["train"]["occurrences"],
            texts=spaces["train"]["texts"],
            occurrence_to_unique=spaces["train"]["occurrence_to_unique"],
            embeddings=embeddings["train"],
            raw_ids=labels,
            embedding_state_signature=json.loads(
                embedding_stage_path.read_text(
                    encoding="utf-8"
                )
            )["metrics"]["embedding_state_signature"],
        )
        _, train_assignments = transform_precomputed_claim_space(
            records["train"], spaces["train"], embeddings["train"], model
        )
        _, val_assignments = transform_precomputed_claim_space(
            records["val"], spaces["val"], embeddings["val"], model
        )
        _, test_assignments = transform_precomputed_claim_space(
            records["test"], spaces["test"], embeddings["test"], model
        )
        if reference is None:
            if seed != reference_seed:
                raise RuntimeError(
                    "Backend override must materialize seed 17 first"
                )
            reference = train_assignments
        train_all = features_from_assignments(
            records["train"],
            train_assignments,
            model,
            encoding=representation["encoding"],
        )
        val_all = features_from_assignments(
            records["val"],
            val_assignments,
            model,
            encoding=representation["encoding"],
        )
        test_all = features_from_assignments(
            records["test"],
            test_assignments,
            model,
            encoding=representation["encoding"],
        )
        ranking = rank_features_by_train_mi(
            train_all,
            encoding=representation["encoding"],
        )
        names = ranking[: int(representation["n_features"])]
        train = train_all[["customer_id", "label", *names]]
        val = val_all[["customer_id", "label", *names]]
        test = test_all[["customer_id", "label", *names]]
        agreement = assignment_agreement(reference, train_assignments)
        summary = {
            "seed": seed,
            "backend": backend,
            "selected_backend": selected_backend,
            "reference_seed": reference_seed,
            "reference_backend": backend,
            "candidate": selected_candidate,
            "n_clusters_after_coverage": len(model["feature_names"]),
            "n_selected_features": len(names),
            "encoding": representation["encoding"],
            "validation_balanced_accuracy": fixed_candidate_score(
                train, val, config
            ),
            **agreement,
        }
        seed_root.mkdir(parents=True, exist_ok=True)
        atomic_frame(seed_root / "cot_features_train.parquet", train)
        atomic_frame(seed_root / "cot_features_val.parquet", val)
        atomic_frame(test_features_path, test)
        atomic_frame(
            assignments_path,
            train_assignments,
        )
        atomic_frame(
            seed_root / "claim_assignments_val.parquet",
            val_assignments,
        )
        atomic_frame(test_assignments_path, test_assignments)
        atomic_write_json(seed_root / "cluster_model.json", model_payload(model))
        atomic_npz(seed_root / "centroids.npz", centroids=model["centroids"])
        atomic_write_json(summary_path, summary)
        rows.append(summary)
    payload = {
        "cell": str(cell),
        "selected_backend": selected_backend,
        "evaluated_backend": backend,
        "reference_seed": reference_seed,
        "additional_seeds": requested_seeds,
        "rows": rows,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("results/v2/derived/reviewer-v4-offline-v2"),
    )
    parser.add_argument(
        "--model",
        choices=("all", "qwen", "gpt_oss"),
        default="all",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(sorted({dataset for dataset, _ in CELLS})),
        default=None,
    )
    parser.add_argument(
        "--backend",
        choices=("selected", "minibatch_kmeans", "agglomerative"),
        default="selected",
        help=(
            "Use the cell's selected backend or rerun in a separate "
            "backend-specific namespace."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 947])
    parser.add_argument(
        "--embedding-cache-cell",
        type=Path,
        help=(
            "Optional compatible cell owning immutable unique-claim E5 "
            "embeddings. Only valid when exactly one dataset/model cell is selected."
        ),
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        type=Path,
        help=(
            "Explicit derived seed_17 cells. This supports isolated/new datasets "
            "without changing the legacy dataset registry."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cells = [
        (dataset, model, args.derived_root / dataset / model / "seed_17")
        for dataset, model in CELLS
        if args.model == "all" or model == args.model
        if args.datasets is None or dataset in args.datasets
    ]
    if args.cells:
        if args.datasets is not None or args.model != "all":
            raise ValueError("--cells cannot be combined with --datasets/--model")
        cells = [(path.parent.parent.name, path.parent.name, path) for path in args.cells]
    backend_override = None if args.backend == "selected" else args.backend
    if args.embedding_cache_cell is not None and len(cells) != 1:
        raise ValueError("--embedding-cache-cell requires exactly one selected cell")
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "reference_seed": 17,
        "additional_seeds": args.seeds,
        "backend": args.backend,
        "cells": [str(cell) for _, _, cell in cells],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    results = []
    for dataset, model, cell in cells:
        results.append(
            run_cell(
                cell,
                args.seeds,
                backend_override=backend_override,
                embedding_cache_cell=args.embedding_cache_cell,
            )
        )
    atomic_write_json(
        args.derived_root
        / f"cluster_seed_stability_{args.model}_{args.backend}.json",
        {"protocol": plan, "results": results},
    )


if __name__ == "__main__":
    main()

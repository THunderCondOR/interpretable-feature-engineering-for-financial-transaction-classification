"""Compare the semantic spaces of two completed feature-selection runs."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from interpretable_features.experiments.artifacts import atomic_write_json


def load_cell(root: Path) -> dict[str, Any]:
    assignment_name = (
        "claim_assignments_train.parquet"
        if (root / "claim_assignments_train.parquet").is_file()
        else "claim_assignments_outer_train.parquet"
    )
    required = (
        "cot_cluster_model.npz",
        "cot_clusters.json",
        "cluster_selection.json",
        assignment_name,
        "cot_features_train.parquet",
        "cot_features_test.parquet",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete offline cell {root}: {missing}")
    clusters = json.loads((root / "cot_clusters.json").read_text())
    meta = clusters.get("cluster_meta", [])
    centroids = np.load(root / "cot_cluster_model.npz")["centroids"].astype(np.float32)
    if len(meta) != len(centroids):
        raise ValueError(f"Cluster/centroid mismatch at {root}: {len(meta)} != {len(centroids)}")
    return {
        "root": root,
        "meta": meta,
        "centroids": centroids,
        "selection": json.loads((root / "cluster_selection.json").read_text()),
        "assignments_train": pd.read_parquet(root / assignment_name),
        "features_train": pd.read_parquet(root / "cot_features_train.parquet"),
        "features_test": pd.read_parquet(root / "cot_features_test.parquet"),
    }


def safe_corr(left: np.ndarray, right: np.ndarray, method: str) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = (
        pearsonr(left, right).statistic if method == "pearson" else spearmanr(left, right).statistic
    )
    return float(value)


def feature_matrix(cell: dict[str, Any], split: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    frame = cell[f"features_{split}"].copy()
    centroid_by_feature = {str(row["feature"]): index for index, row in enumerate(cell["meta"])}
    # The train-only representation selector may retain only a subset of the
    # semantic clusters.  Compare exactly the materialized columns and keep an
    # explicit mapping back to the full centroid matrix.
    features = [str(name) for name in frame.columns if str(name) in centroid_by_feature]
    if not features:
        raise ValueError(f"No semantic feature columns in {cell['root']}")
    indices = np.asarray([centroid_by_feature[name] for name in features], dtype=int)
    return frame.set_index("customer_id"), features, indices


def client_semantic_vectors(binary: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    vectors = binary.astype(np.float32) @ centroids
    counts = binary.sum(axis=1, keepdims=True)
    vectors /= np.maximum(counts, 1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= np.maximum(norms, 1e-12)
    return vectors


def compare_cells(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_model: str,
    right_model: str,
    left_seed: int,
    right_seed: int,
    comparison_axis: str,
) -> dict[str, Any]:
    similarity = cosine_similarity(left["centroids"], right["centroids"])
    left_idx, right_idx = linear_sum_assignment(-similarity)
    matched_cosine = similarity[left_idx, right_idx]
    threshold_mask = matched_cosine >= 0.70
    left_weights = np.asarray(
        [row.get("assigned_occurrences", row.get("occurrences", 1)) for row in left["meta"]],
        dtype=float,
    )
    right_weights = np.asarray(
        [row.get("assigned_occurrences", row.get("occurrences", 1)) for row in right["meta"]],
        dtype=float,
    )
    left_weights /= max(left_weights.sum(), 1)
    right_weights /= max(right_weights.sum(), 1)

    left_assign = (
        left["assignments_train"]
        .sort_values("assignment_distance")
        .drop_duplicates("normalized_text")
    )
    right_assign = (
        right["assignments_train"]
        .sort_values("assignment_distance")
        .drop_duplicates("normalized_text")
    )
    common = left_assign[["normalized_text", "cluster_id"]].merge(
        right_assign[["normalized_text", "cluster_id"]],
        on="normalized_text",
        suffixes=("_left", "_right"),
    )
    exact_ari = (
        adjusted_rand_score(common.cluster_id_left, common.cluster_id_right)
        if len(common) >= 2
        else None
    )
    exact_nmi = (
        normalized_mutual_info_score(common.cluster_id_left, common.cluster_id_right)
        if len(common) >= 2
        else None
    )

    left_test, left_features, left_feature_indices = feature_matrix(left, "test")
    right_test, right_features, right_feature_indices = feature_matrix(right, "test")
    clients = left_test.index.intersection(right_test.index).sort_values()
    if clients.empty:
        raise ValueError("Feature sets have no common test clients")
    left_binary = left_test.loc[clients, left_features].to_numpy() > 0
    right_binary = right_test.loc[clients, right_features].to_numpy() > 0
    left_vectors = client_semantic_vectors(left_binary, left["centroids"][left_feature_indices])
    right_vectors = client_semantic_vectors(right_binary, right["centroids"][right_feature_indices])
    client_cosine = np.sum(left_vectors * right_vectors, axis=1)

    paired_left, paired_right = left_idx[threshold_mask], right_idx[threshold_mask]
    left_selected_position = {
        centroid_index: position for position, centroid_index in enumerate(left_feature_indices)
    }
    right_selected_position = {
        centroid_index: position for position, centroid_index in enumerate(right_feature_indices)
    }
    selected_pairs = [
        (left_selected_position[a], right_selected_position[b])
        for a, b in zip(paired_left, paired_right)
        if a in left_selected_position and b in right_selected_position
    ]
    if selected_pairs:
        selected_left, selected_right = map(np.asarray, zip(*selected_pairs))
        pair_a = left_binary[:, selected_left]
        pair_b = right_binary[:, selected_right]
    else:
        pair_a = np.zeros((len(clients), 0), dtype=bool)
        pair_b = np.zeros((len(clients), 0), dtype=bool)
    intersections = np.logical_and(pair_a, pair_b).sum(axis=1)
    unions = np.logical_or(pair_a, pair_b).sum(axis=1)
    client_jaccard = np.divide(
        intersections, unions, out=np.ones_like(intersections, dtype=float), where=unions > 0
    )
    prevalence_left, prevalence_right = pair_a.mean(0), pair_b.mean(0)

    left_train, left_train_features, left_train_indices = feature_matrix(left, "train")
    right_train, right_train_features, right_train_indices = feature_matrix(right, "train")
    train_clients = left_train.index.intersection(right_train.index).sort_values()
    labels_left = left_train.loc[train_clients, "label"].to_numpy()
    labels_right = right_train.loc[train_clients, "label"].to_numpy()
    if not np.array_equal(labels_left, labels_right):
        raise ValueError("Train labels/client ordering changed across generation seeds")
    prior = labels_left.mean()
    lift_left, lift_right = [], []
    left_train_name = dict(zip(left_train_indices, left_train_features))
    right_train_name = dict(zip(right_train_indices, right_train_features))
    for a, b in zip(paired_left, paired_right):
        if a not in left_train_name or b not in right_train_name:
            continue
        active_a = left_train.loc[train_clients, left_train_name[a]].to_numpy() > 0
        active_b = right_train.loc[train_clients, right_train_name[b]].to_numpy() > 0
        lift_left.append(labels_left[active_a].mean() - prior if active_a.any() else 0.0)
        lift_right.append(labels_right[active_b].mean() - prior if active_b.any() else 0.0)

    return {
        "comparison_axis": comparison_axis,
        "left_model": left_model,
        "right_model": right_model,
        "left_seed": left_seed,
        "right_seed": right_seed,
        "left_clusters": len(left["meta"]),
        "right_clusters": len(right["meta"]),
        "left_selected_candidate": left["selection"].get("selected_candidate"),
        "right_selected_candidate": right["selection"].get("selected_candidate"),
        "hungarian_pairs": len(left_idx),
        "hungarian_mean_cosine": float(matched_cosine.mean()),
        "matched_pairs_cosine_ge_0_70": int(threshold_mask.sum()),
        "matched_pair_share_ge_0_70": float(threshold_mask.mean()),
        "matched_left_occurrence_mass_ge_0_70": float(left_weights[paired_left].sum()),
        "matched_right_occurrence_mass_ge_0_70": float(right_weights[paired_right].sum()),
        "exact_common_claims": len(common),
        "exact_common_claim_coverage_left": len(common)
        / max(left_assign.normalized_text.nunique(), 1),
        "exact_common_claim_coverage_right": len(common)
        / max(right_assign.normalized_text.nunique(), 1),
        "exact_anchor_ari": float(exact_ari) if exact_ari is not None else None,
        "exact_anchor_nmi": float(exact_nmi) if exact_nmi is not None else None,
        "shared_test_clients": len(clients),
        "matched_selected_feature_pairs": len(selected_pairs),
        "client_semantic_cosine_mean": float(client_cosine.mean()),
        "client_semantic_cosine_median": float(np.median(client_cosine)),
        "client_semantic_cosine_p10": float(np.quantile(client_cosine, 0.10)),
        "client_matched_feature_jaccard_mean": float(client_jaccard.mean()),
        "client_matched_feature_jaccard_median": float(np.median(client_jaccard)),
        "matched_cluster_prevalence_pearson": safe_corr(
            prevalence_left, prevalence_right, "pearson"
        ),
        "matched_cluster_prevalence_spearman": safe_corr(
            prevalence_left, prevalence_right, "spearman"
        ),
        "matched_cluster_label_lift_pearson": safe_corr(
            np.asarray(lift_left), np.asarray(lift_right), "pearson"
        ),
        "matched_cluster_label_lift_spearman": safe_corr(
            np.asarray(lift_left), np.asarray(lift_right), "spearman"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument(
        "--axis", required=True, choices=["generation_seed", "clustering_seed", "model"]
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    runs = [
        json.loads((root / "source_manifest.json").read_text())["source_contract"]
        for root in (args.left, args.right)
    ]
    if runs[0]["dataset"] != runs[1]["dataset"]:
        parser.error("Both runs must use the same dataset")
    comparison = compare_cells(
        load_cell(args.left),
        load_cell(args.right),
        comparison_axis=args.axis,
        left_model=runs[0]["model_slug"],
        right_model=runs[1]["model_slug"],
        left_seed=runs[0]["seed"],
        right_seed=runs[1]["seed"],
    )
    atomic_write_json(args.output, comparison)


if __name__ == "__main__":
    main()

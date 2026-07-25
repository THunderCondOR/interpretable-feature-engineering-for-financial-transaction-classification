import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from src.experiments.derived_artifacts import (
    compatible_stage,
    complete_stage,
    stage_identity,
)
from src.pipeline.semantic_features import (
    cut_agglomerative_hierarchy,
    fit_agglomerative_hierarchy,
    nearest_centroid_assignments,
    normalize_claim,
    unique_claim_space,
)
from scripts.run_v4_offline_pipeline import (
    choose_candidate,
    choose_clustering_backend,
    choose_representation,
    compatible_candidates,
    features_from_assignments,
)


def test_hierarchy_fixed_k_cut_matches_sklearn_partition():
    values = np.asarray([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
        [0.01, 0.99],
        [-1.0, 0.0],
        [-0.99, 0.01],
    ], dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    hierarchy = fit_agglomerative_hierarchy(values)
    observed = cut_agglomerative_hierarchy(hierarchy, n_clusters=3)
    expected = AgglomerativeClustering(
        n_clusters=3,
        metric="cosine",
        linkage="average",
    ).fit_predict(values)
    assert adjusted_rand_score(expected, observed) == 1.0


def test_hierarchy_threshold_cut_matches_sklearn_partition():
    values = np.asarray([
        [1.0, 0.0],
        [0.98, 0.02],
        [0.0, 1.0],
        [0.02, 0.98],
    ], dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    threshold = 0.01
    hierarchy = fit_agglomerative_hierarchy(values)
    observed = cut_agglomerative_hierarchy(
        hierarchy,
        distance_threshold=threshold,
    )
    expected = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(values)
    assert adjusted_rand_score(expected, observed) == 1.0


def test_unique_claim_space_deduplicates_types_but_keeps_occurrences():
    records = [
        {
            "customer_id": 1,
            "label": 0,
            "claim_records": [
                {"claim_id": "a", "original_text": "The client uses cash."},
                {"claim_id": "b", "original_text": "The client uses cash."},
            ],
        },
        {
            "customer_id": 2,
            "label": 1,
            "claim_records": [
                {"claim_id": "c", "original_text": "The client shops online."},
            ],
        },
    ]
    space = unique_claim_space(records)
    assert len(space["occurrences"]) == 3
    assert len(space["texts"]) == 2
    assert space["occurrence_to_unique"].tolist() == [0, 0, 1]


def test_claim_normalization_preserves_domain_bearing_words():
    text = "  The client conducted a HIGH volume of transactions. "
    assert normalize_claim(text) == (
        "the client conducted a high volume of transactions"
    )
    space = unique_claim_space([{
        "customer_id": 1,
        "label": 0,
        "claim_records": [{
            "claim_id": "legacy",
            "original_text": text,
            "normalized_text": "the client conducted a high volume of",
        }],
    }])
    assert space["texts"] == [
        "the client conducted a high volume of transactions"
    ]


def test_candidate_selection_uses_simplest_within_tie_margin():
    selected = choose_candidate([
        {
            "candidate": "threshold_0.01",
            "validation_balanced_accuracy": 0.705,
            "n_clusters": 3000,
        },
        {
            "candidate": "k_400",
            "validation_balanced_accuracy": 0.702,
            "n_clusters": 400,
        },
        {
            "candidate": "k_200",
            "validation_balanced_accuracy": 0.699,
            "n_clusters": 200,
        },
    ], 0.005)
    assert selected["candidate"] == "k_400"


def test_nearest_centroid_assignment_respects_batch_boundaries():
    embeddings = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.99, 0.01],
        [0.01, 0.99],
    ], dtype=np.float32)
    centroids = np.eye(2, dtype=np.float32)
    assignments, distances = nearest_centroid_assignments(
        embeddings,
        centroids,
        max_distance=0.1,
        batch_size=2,
    )
    assert assignments.tolist() == [0, 1, 0, 1]
    assert np.all(distances < 0.1)


def test_completed_stage_rejects_modified_output(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"value": 1}', encoding="utf-8")
    identity = stage_identity(
        stage="fixture",
        source={"source_root": "fixture"},
        inputs={"input": 1},
        configuration={"setting": 2},
    )
    manifest = tmp_path / "stage.json"
    complete_stage(manifest, identity, outputs=[output])
    assert compatible_stage(manifest, identity)

    output.write_text('{"value": 2}', encoding="utf-8")
    assert not compatible_stage(manifest, identity)


def test_auto_backend_avoids_quadratic_clustering_for_large_claim_space():
    assert choose_clustering_backend(45_000, "auto") == "agglomerative"
    assert choose_clustering_backend(45_001, "auto") == "minibatch_kmeans"
    assert (
        choose_clustering_backend(100_000, "agglomerative")
        == "agglomerative"
    )


def test_scalable_backend_keeps_only_fixed_cluster_candidates():
    assert compatible_candidates(
        ["threshold_0.01", "k_200", "k_400", "k_800"],
        "minibatch_kmeans",
    ) == ["k_200", "k_400", "k_800"]


def test_representation_selection_prefers_smaller_space_within_tie():
    selected = choose_representation([
        {
            "encoding": "raw_count",
            "n_features": 150,
            "validation_balanced_accuracy": 0.651,
        },
        {
            "encoding": "binary",
            "n_features": 100,
            "validation_balanced_accuracy": 0.647,
        },
        {
            "encoding": "binary",
            "n_features": 50,
            "validation_balanced_accuracy": 0.64,
        },
    ], tie_margin=0.005)
    assert selected["encoding"] == "binary"
    assert selected["n_features"] == 100


def test_assignment_features_support_binary_count_and_normalized():
    records = [
        {"customer_id": 10, "label": 0},
        {"customer_id": 20, "label": 1},
    ]
    assignments = pd.DataFrame([
        {"customer_id": 10, "cluster_index": 0, "assigned": True},
        {"customer_id": 10, "cluster_index": 0, "assigned": True},
        {"customer_id": 10, "cluster_index": 1, "assigned": True},
        {"customer_id": 20, "cluster_index": -1, "assigned": False},
    ])
    model = {"feature_names": ["cot_a", "cot_b"]}
    binary = features_from_assignments(
        records, assignments, model, encoding="binary"
    )
    raw = features_from_assignments(
        records, assignments, model, encoding="raw_count"
    )
    normalized = features_from_assignments(
        records, assignments, model, encoding="normalized_count"
    )
    assert binary[["cot_a", "cot_b"]].to_numpy().tolist() == [
        [1.0, 1.0], [0.0, 0.0]
    ]
    assert raw[["cot_a", "cot_b"]].to_numpy().tolist() == [
        [2.0, 1.0], [0.0, 0.0]
    ]
    assert np.allclose(
        normalized[["cot_a", "cot_b"]].to_numpy(),
        [[2 / 3, 1 / 3], [0.0, 0.0]],
    )

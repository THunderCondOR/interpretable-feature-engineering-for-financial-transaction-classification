import numpy as np
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
    unique_claim_space,
)
from scripts.run_v4_offline_pipeline import choose_candidate


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

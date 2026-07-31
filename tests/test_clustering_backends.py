import numpy as np

from scripts.run_rosbank_clustering_sweep import jobs, summarize
from src.pipeline.clustering_backends import (
    fit_hdbscan_pca,
    fit_spherical_kmeans,
    parse_hdbscan_candidate,
)


def test_hdbscan_candidate_parser():
    assert parse_hdbscan_candidate("pca64_mcs25_ms5") == {
        "dimensions": 64,
        "min_cluster_size": 25,
        "min_samples": 5,
    }


def test_spherical_kmeans_recovers_separated_unit_groups():
    rng = np.random.default_rng(17)
    values = np.vstack([
        np.array([1.0, 0.0]) + rng.normal(0, 0.02, (20, 2)),
        np.array([0.0, 1.0]) + rng.normal(0, 0.02, (20, 2)),
    ]).astype(np.float32)
    labels, metadata = fit_spherical_kmeans(
        values, n_clusters=2, seed=17, device="cpu"
    )
    assert len(set(labels[:20])) == 1
    assert len(set(labels[20:])) == 1
    assert labels[0] != labels[-1]
    assert metadata["algorithm"] == "spherical_kmeans_cosine"


def test_hdbscan_preserves_noise_label():
    rng = np.random.default_rng(101)
    values = np.vstack([
        rng.normal(-2, 0.05, (30, 6)),
        rng.normal(2, 0.05, (30, 6)),
        np.array([[20, -20, 20, -20, 20, -20]]),
    ]).astype(np.float32)
    labels, metadata = fit_hdbscan_pca(
        values,
        dimensions=4,
        min_cluster_size=10,
        min_samples=5,
        seed=17,
    )
    assert len(labels) == len(values)
    assert metadata["algorithm"] == "hdbscan_train_pca"
    assert set(labels) - {-1}


def test_rosbank_sweep_has_reference_and_three_scalable_families(tmp_path):
    queue = jobs(tmp_path)
    backends = {job["backend"] for job in queue}
    assert backends == {
        "minibatch_kmeans",
        "spherical_kmeans",
        "hdbscan_pca",
        "agglomerative",
    }
    assert len([job for job in queue if job["backend"] == "agglomerative"]) == 2


def test_sweep_selection_uses_mean_across_models_and_seeds():
    rows = []
    scores = {
        "minibatch_kmeans": (0.65, 0.66),
        "spherical_kmeans": (0.66, 0.67),
        "hdbscan_pca": (0.60, 0.61),
    }
    for backend, model_scores in scores.items():
        for model, score in zip(("qwen", "gpt_oss"), model_scores):
            for seed in (17, 101, 947):
                rows.append({
                    "backend": backend,
                    "model": model,
                    "seed": seed,
                    "validation_balanced_accuracy": score,
                })
    selection, _ = summarize(rows)
    assert selection["selected_backend"] == "spherical_kmeans"

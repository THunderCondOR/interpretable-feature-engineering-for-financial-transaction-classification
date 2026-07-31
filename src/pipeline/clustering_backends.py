"""Scalable label-agnostic clustering backends for normalized claim embeddings."""

from __future__ import annotations

import re

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


HDBSCAN_PATTERN = re.compile(
    r"^pca(?P<dimensions>\d+)_mcs(?P<min_cluster_size>\d+)_ms(?P<min_samples>\d+)$"
)


def parse_hdbscan_candidate(candidate: str) -> dict[str, int]:
    match = HDBSCAN_PATTERN.fullmatch(candidate)
    if not match:
        raise ValueError(f"Invalid HDBSCAN candidate: {candidate}")
    return {key: int(value) for key, value in match.groupdict().items()}


def fit_hdbscan_pca(
    embeddings: np.ndarray,
    *,
    dimensions: int,
    min_cluster_size: int,
    min_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    values = np.asarray(embeddings, dtype=np.float32)
    effective_dimensions = min(int(dimensions), values.shape[1], len(values) - 1)
    if effective_dimensions < 2:
        raise ValueError("HDBSCAN/PCA requires at least two effective dimensions")
    projected = PCA(
        n_components=effective_dimensions,
        svd_solver="randomized",
        random_state=int(seed),
    ).fit_transform(values)
    projected = normalize(projected).astype(np.float32, copy=False)
    labels = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        n_jobs=-1,
        copy=True,
    ).fit_predict(projected).astype(np.int32)
    return labels, {
        "algorithm": "hdbscan_train_pca",
        "pca_dimensions": effective_dimensions,
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "noise_claims": int((labels < 0).sum()),
    }


def fit_spherical_kmeans(
    embeddings: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    max_iter: int = 50,
    tolerance: float = 1e-4,
    batch_size: int = 4096,
    device: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Cluster unit vectors with cosine assignment and normalized centroids."""
    import torch

    values_np = normalize(
        np.asarray(embeddings, dtype=np.float32)
    ).astype(np.float32, copy=False)
    count = len(values_np)
    clusters = min(int(n_clusters), count)
    if clusters < 1:
        raise ValueError("Spherical KMeans requires at least one cluster")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    values = torch.as_tensor(values_np, device=selected_device)
    generator = torch.Generator(device=selected_device)
    generator.manual_seed(int(seed))
    initial = torch.randperm(count, generator=generator, device=selected_device)[
        :clusters
    ]
    centroids = values[initial].clone()
    centroids = torch.nn.functional.normalize(centroids, dim=1)
    labels = torch.empty(count, dtype=torch.long, device=selected_device)
    iterations = 0
    objective = float("-inf")

    for iteration in range(int(max_iter)):
        similarities = []
        for start in range(0, count, int(batch_size)):
            stop = min(start + int(batch_size), count)
            similarities.append(values[start:stop] @ centroids.T)
        joined = torch.cat(similarities, dim=0)
        best_similarity, new_labels = joined.max(dim=1)
        new_centroids = torch.zeros_like(centroids)
        new_centroids.index_add_(0, new_labels, values)
        cluster_sizes = torch.bincount(new_labels, minlength=clusters)
        empty = cluster_sizes == 0
        if empty.any():
            replacements = torch.topk(
                -best_similarity, int(empty.sum().item())
            ).indices
            new_centroids[empty] = values[replacements]
        new_centroids = torch.nn.functional.normalize(new_centroids, dim=1)
        change = float((1.0 - (centroids * new_centroids).sum(1)).abs().max())
        centroids = new_centroids
        labels = new_labels
        objective = float(best_similarity.mean())
        iterations = iteration + 1
        del joined, best_similarity
        if change <= float(tolerance):
            break

    result = labels.detach().cpu().numpy().astype(np.int32)
    return result, {
        "algorithm": "spherical_kmeans_cosine",
        "n_clusters": clusters,
        "iterations": iterations,
        "mean_cosine_similarity": objective,
        "device": selected_device,
    }

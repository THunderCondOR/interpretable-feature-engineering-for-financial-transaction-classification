"""Build CoT cluster features without validation/test leakage.

The clustering model is fitted only on train claims. Validation and test claims are
assigned to the nearest retained train cluster. Labels from validation/test are
kept only for downstream evaluation and are never used in feature construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from src.utils.cluster import embed_texts
from src.utils.filtration import clean_text, filter_cluster_ids_by_class_diff, label_based_outlier_detection

SPLITS = ("train", "val", "test")


def split_output_path(config: dict, key: str, split: str) -> Path:
    out_dir = Path(config["output"]["base_dir"])
    base = Path(config["output"][key])
    return out_dir / f"{base.stem}_{split}{base.suffix}"


def load_claim_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def flatten_claim_records(records: list[dict[str, Any]]) -> tuple[list[str], list[int], list[int]]:
    claims: list[str] = []
    labels: list[int] = []
    customer_ids: list[int] = []
    for record in records:
        for claim in record.get("claims", []):
            text = clean_text(str(claim))
            if text:
                claims.append(text)
                labels.append(int(record["label"]))
                customer_ids.append(int(record["customer_id"]))
    return claims, labels, customer_ids


def fit_train_clusters(config: dict, train_records: list[dict[str, Any]]) -> dict[str, Any]:
    claims, labels, customer_ids = flatten_claim_records(train_records)
    if not claims:
        raise ValueError("No train claims found. Run claim extraction before clustering.")

    embedding_model = config["pipeline"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embed_texts(claims, model_name=embedding_model)

    top_k = min(5, max(len(claims) - 1, 1))
    outlier_flags = label_based_outlier_detection(embeddings, labels, top_k=top_k) if len(claims) > 2 else [False] * len(claims)
    keep_mask = np.asarray([not flag for flag in outlier_flags], dtype=bool)
    embeddings = embeddings[keep_mask]
    claims = [claim for claim, keep in zip(claims, keep_mask) if keep]
    labels = [label for label, keep in zip(labels, keep_mask) if keep]
    customer_ids = [cid for cid, keep in zip(customer_ids, keep_mask) if keep]

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(config["pipeline"].get("distance_threshold", 0.01)),
        metric="cosine",
        linkage="average",
    )
    raw_cluster_ids = clusterer.fit_predict(embeddings)
    cluster_ids = filter_cluster_ids_by_class_diff(
        raw_cluster_ids,
        labels,
        int(config["dataset"]["num_labels"]),
        min_diff=float(config["pipeline"].get("class_diff_threshold", 0.02)),
    )

    min_cluster_size = int(config["pipeline"].get("min_cluster_size", 1))
    kept_clusters = [
        cluster_id
        for cluster_id in sorted(set(int(c) for c in cluster_ids if c >= 0))
        if int(np.sum(cluster_ids == cluster_id)) >= min_cluster_size
    ]
    if not kept_clusters:
        raise ValueError("No CoT clusters survived filtering. Lower thresholds or inspect claims.")

    centroids = []
    feature_names = []
    cluster_meta = []
    for new_idx, old_cluster_id in enumerate(kept_clusters):
        mask = cluster_ids == old_cluster_id
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
        feature_name = f"cot_cluster_{new_idx:04d}"
        feature_names.append(feature_name)
        cluster_claims = [claim for claim, keep in zip(claims, mask) if keep]
        cluster_labels = [label for label, keep in zip(labels, mask) if keep]
        label_counts = pd.Series(cluster_labels).value_counts().sort_index().to_dict()
        cluster_meta.append(
            {
                "feature": feature_name,
                "old_cluster_id": int(old_cluster_id),
                "size": int(mask.sum()),
                "label_counts": {str(k): int(v) for k, v in label_counts.items()},
                "examples": cluster_claims[:10],
            }
        )

    return {
        "centroids": np.vstack(centroids),
        "feature_names": feature_names,
        "cluster_meta": cluster_meta,
    }


def records_to_features(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    customer_ids: list[int],
    model: dict[str, Any],
    max_distance: float,
) -> pd.DataFrame:
    feature_names = model["feature_names"]
    vectors = {int(record["customer_id"]): np.zeros(len(feature_names), dtype=np.float32) for record in records}

    if len(customer_ids):
        distances = cosine_distances(embeddings, model["centroids"])
        nearest = distances.argmin(axis=1)
        nearest_distance = distances.min(axis=1)
        for cid, cluster_idx, distance in zip(customer_ids, nearest, nearest_distance):
            if float(distance) <= max_distance:
                vectors[int(cid)][int(cluster_idx)] += 1.0

    rows = []
    for record in records:
        cid = int(record["customer_id"])
        row = {"customer_id": cid, "label": int(record["label"])}
        row.update({name: float(value) for name, value in zip(feature_names, vectors[cid])})
        rows.append(row)
    return pd.DataFrame(rows)


def transform_split(config: dict, records: list[dict[str, Any]], model: dict[str, Any]) -> pd.DataFrame:
    claims, _labels, customer_ids = flatten_claim_records(records)
    embedding_model = config["pipeline"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embed_texts(claims, model_name=embedding_model) if claims else np.zeros((0, model["centroids"].shape[1]))
    return records_to_features(
        records,
        embeddings,
        customer_ids,
        model,
        max_distance=float(config["pipeline"].get("max_assign_distance", 0.45)),
    )


def build_cot_features(config: dict) -> None:
    out_dir = Path(config["output"]["base_dir"])
    train_records = load_claim_records(split_output_path(config, "claims", "train"))
    model = fit_train_clusters(config, train_records)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "cot_clusters.json"
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(model["cluster_meta"], file, indent=2, ensure_ascii=False)
    print(f"Saved CoT cluster metadata -> {meta_path}")

    for split in SPLITS:
        records = load_claim_records(split_output_path(config, "claims", split))
        features = transform_split(config, records, model)
        path = out_dir / f"cot_features_{split}.parquet"
        features.to_parquet(path, index=False)
        print(f"Saved {split} CoT features: shape={features.shape} -> {path}")

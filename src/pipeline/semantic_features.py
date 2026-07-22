"""Label-agnostic semantic clustering with frozen train-derived transforms."""
from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import cosine_distances

from src.experiments.artifacts import fingerprint
from src.utils.cluster import embed_texts
from src.utils.filtration import clean_text

Embedder = Callable[..., np.ndarray]


def normalize_claim(text: str) -> str:
    return " ".join(clean_text(str(text)).lower().split()).strip(" .")


def claim_occurrences(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        entries = record.get("claim_records") or [
            {"original_text": item} for item in record.get("claims", [])
        ]
        for index, entry in enumerate(entries):
            original = str(entry.get("original_text", entry.get("text", ""))).strip()
            normalized = str(entry.get("normalized_text") or normalize_claim(original))
            if normalized:
                rows.append({
                    "claim_id": entry.get("claim_id") or fingerprint({
                        "customer_id": record["customer_id"], "text": normalized, "index": index,
                    })[:20],
                    "customer_id": int(record["customer_id"]),
                    "label": int(record.get("label", -1)),
                    "original_text": original,
                    "normalized_text": normalized,
                })
    return rows


def _unique(occurrences):
    texts, indices, mapping = [], {}, []
    for row in occurrences:
        text = row["normalized_text"]
        if text not in indices:
            indices[text] = len(texts)
            texts.append(text)
        mapping.append(indices[text])
    return texts, np.asarray(mapping, dtype=np.int32)


def _normal(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def _settings(config):
    legacy, explicit = config.get("pipeline", {}), config.get("clustering", {})
    return {
        "distance_threshold": float(explicit.get("distance_threshold", legacy.get("distance_threshold", 0.01))),
        "n_clusters": explicit.get("n_clusters"),
        "min_client_coverage": int(explicit.get("min_client_coverage", legacy.get("min_cluster_size", 5))),
        "max_train_distance": float(explicit.get("max_train_distance", 1.0)),
        "max_assign_distance": float(explicit.get("max_assign_distance", legacy.get("max_assign_distance", 0.45))),
        "feature_encoding": explicit.get("feature_encoding", "binary"),
    }


def _cluster(embeddings, settings):
    if len(embeddings) == 1:
        return np.zeros(1, dtype=np.int32)
    kwargs = {"metric": "cosine", "linkage": "average"}
    if settings["n_clusters"] is None:
        kwargs.update(n_clusters=None, distance_threshold=settings["distance_threshold"])
    else:
        kwargs.update(n_clusters=min(int(settings["n_clusters"]), len(embeddings)))
    return AgglomerativeClustering(**kwargs).fit_predict(embeddings).astype(np.int32)


def _centroid(values):
    center = values.mean(axis=0)
    return center / max(float(np.linalg.norm(center)), 1e-12)


def fit_semantic_space(config, train_records, *, embedder: Embedder = embed_texts):
    """Fit from train texts only. Labels below are post-hoc metadata."""
    occurrences = claim_occurrences(train_records)
    if not occurrences:
        raise ValueError("No train claims found")
    texts, occurrence_to_unique = _unique(occurrences)
    model_name = config.get("clustering", {}).get(
        "embedding_model", config.get("pipeline", {}).get("embedding_model", "tf-idf")
    )
    embeddings = _normal(embedder(texts, model_name=model_name))
    settings = _settings(config)
    raw_ids = _cluster(embeddings, settings)
    centroids, metadata = [], []
    for raw_id in sorted(set(raw_ids.tolist())):
        unique_indices = np.flatnonzero(raw_ids == raw_id)
        center = _centroid(embeddings[unique_indices])
        initial = cosine_distances(embeddings[unique_indices], center.reshape(1, -1)).ravel()
        accepted = unique_indices[initial <= settings["max_train_distance"]]
        occurrence_indices = np.flatnonzero(np.isin(occurrence_to_unique, accepted))
        clients = {occurrences[i]["customer_id"] for i in occurrence_indices}
        if len(clients) < settings["min_client_coverage"]:
            continue
        center = _centroid(embeddings[accepted])
        distances = cosine_distances(embeddings[accepted], center.reshape(1, -1)).ravel()
        medoid_index = int(accepted[int(distances.argmin())])
        cluster_id = f"clu_{fingerprint(texts[medoid_index])[:12]}"
        labels = [occurrences[i]["label"] for i in occurrence_indices]
        centroids.append(center)
        metadata.append({
            "cluster_id": cluster_id, "feature": f"cot_{cluster_id}",
            "medoid": texts[medoid_index],
            "examples": [texts[i] for i in accepted[np.argsort(distances)[:10]]],
            "unique_claims": int(len(accepted)), "occurrences": int(len(occurrence_indices)),
            "unique_clients": int(len(clients)),
            "compactness_mean_distance": float(distances.mean()),
            "assignment_distance_p95": float(np.quantile(distances, 0.95)),
            "label_counts": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        })
    if not centroids:
        raise ValueError("No semantic clusters meet min_client_coverage")
    model = {
        "centroids": np.vstack(centroids),
        "feature_names": [row["feature"] for row in metadata],
        "cluster_meta": metadata, "embedding_model": model_name, "settings": settings,
        "formation_signature": fingerprint({"texts": texts, "model": model_name, "settings": settings}),
    }
    model.update(fit_supervised_selection(config, transform_semantic_space(config, train_records, model, embedder=embedder)))
    return model


def transform_semantic_space(config, records, model, *, embedder: Embedder = embed_texts):
    """Assign any split to frozen train centroids without consulting labels."""
    occurrences = claim_occurrences(records)
    texts, occurrence_to_unique = _unique(occurrences)
    vectors = {
        int(row["customer_id"]): np.zeros(len(model["feature_names"]), dtype=np.float32)
        for row in records
    }
    if texts:
        embeddings = _normal(embedder(texts, model_name=model["embedding_model"]))
        distances = cosine_distances(embeddings, model["centroids"])
        nearest = distances.argmin(axis=1)
        accepted = distances.min(axis=1) <= model["settings"]["max_assign_distance"]
        for occurrence_index, row in enumerate(occurrences):
            unique_index = occurrence_to_unique[occurrence_index]
            if accepted[unique_index]:
                vectors[row["customer_id"]][int(nearest[unique_index])] += 1.0
    encoding, rows = model["settings"]["feature_encoding"], []
    for record in records:
        cid, values = int(record["customer_id"]), vectors[int(record["customer_id"])]
        if encoding == "binary":
            values = (values > 0).astype(np.float32)
        elif encoding == "normalized_count":
            values = values / max(float(values.sum()), 1.0)
        elif encoding != "raw_count":
            raise ValueError(f"Unknown feature_encoding: {encoding}")
        row = {"customer_id": cid, "label": int(record.get("label", -1))}
        row.update(dict(zip(model["feature_names"], values.astype(float))))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_supervised_selection(config, train_features):
    """Optional and explicit train-only supervised selection."""
    selection = config.get("feature_selection", {})
    names = [column for column in train_features if column.startswith("cot_")]
    if not selection.get("enabled", False):
        return {"selected_feature_names": names, "selection_scores": {}, "selection_mode": "disabled"}
    scores = mutual_info_classif(
        train_features[names].to_numpy(), train_features["label"].to_numpy(),
        discrete_features=True, random_state=int(selection.get("seed", 17)),
    )
    ranked = sorted(zip(names, scores), key=lambda item: (-item[1], item[0]))
    selected = [name for name, score in ranked if float(score) > float(selection.get("min_mutual_information", 0.0))]
    if selection.get("max_features"):
        selected = selected[: int(selection["max_features"])]
    return {
        "selected_feature_names": selected,
        "selection_scores": {name: float(score) for name, score in ranked},
        "selection_mode": "train_only_mutual_information",
    }


def selected_view(frame, model):
    return frame[["customer_id", "label", *model.get("selected_feature_names", [])]].copy()

"""Label-agnostic semantic clustering with frozen train-derived transforms."""
from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.pairwise import cosine_distances

from src.experiments.artifacts import fingerprint
from src.utils.cluster import embed_texts, embedding_input_prefix
from src.data.entity_ids import canonical_entity_id

Embedder = Callable[..., np.ndarray]


def normalize_claim(text: str) -> str:
    """Normalize formatting without deleting domain-bearing words."""
    return " ".join(str(text).casefold().split()).strip(" .")


def claim_occurrences(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        entries = record.get("claim_records") or [
            {"original_text": item} for item in record.get("claims", [])
        ]
        for index, entry in enumerate(entries):
            original = str(entry.get("original_text", entry.get("text", ""))).strip()
            # Recompute from the immutable original text.  Older API artifacts
            # contain a destructive normalized_text field which removed words
            # such as "transaction(s)" and must not define the semantic space.
            normalized = normalize_claim(original)
            if normalized:
                rows.append({
                    "claim_id": entry.get("claim_id") or fingerprint({
                        "customer_id": record["customer_id"], "text": normalized, "index": index,
                    })[:20],
                    "customer_id": canonical_entity_id(record["customer_id"]),
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


def unique_claim_space(records):
    """Return the canonical atomic-claim space used by every clustering runner."""
    occurrences = claim_occurrences(records)
    if not occurrences:
        raise ValueError("No claims found")
    texts, occurrence_to_unique = _unique(occurrences)
    return {
        "occurrences": occurrences,
        "texts": texts,
        "occurrence_to_unique": occurrence_to_unique,
    }


def _normal(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def fit_text_embedding_space(texts, model_name, *, embedder: Embedder = embed_texts):
    """Fit a text representation on train texts and return its frozen state."""
    if model_name == "tf-idf":
        transformer = TfidfVectorizer(
            max_features=384,
            stop_words=None,
            lowercase=True,
        )
        embeddings = transformer.fit_transform(texts).toarray()
        signature = fingerprint({
            "vocabulary": transformer.vocabulary_,
            "idf": transformer.idf_.tolist(),
        })
        return _normal(embeddings), transformer, signature
    embeddings = _normal(embedder(texts, model_name=model_name))
    return embeddings, None, fingerprint({
        "model_name": model_name,
        "input_prefix": embedding_input_prefix(model_name),
    })


def transform_text_embedding_space(
    texts,
    model_name,
    transformer=None,
    *,
    embedder: Embedder = embed_texts,
):
    """Transform with the train-fitted representation."""
    if model_name == "tf-idf":
        if transformer is None:
            raise ValueError("Frozen TF-IDF transformer is required for split assignment")
        return _normal(transformer.transform(texts).toarray())
    return _normal(embedder(texts, model_name=model_name))


def _settings(config):
    legacy, explicit = config.get("pipeline", {}), config.get("clustering", {})
    settings = {
        "distance_threshold": float(explicit.get("distance_threshold", legacy.get("distance_threshold", 0.01))),
        "n_clusters": explicit.get("n_clusters"),
        "min_client_coverage": int(explicit.get("min_client_coverage", legacy.get("min_cluster_size", 5))),
        "max_train_distance": float(explicit.get("max_train_distance", 1.0)),
        "max_assign_distance": float(explicit.get("max_assign_distance", legacy.get("max_assign_distance", 0.45))),
        "assignment_quantile": (
            float(explicit["assignment_quantile"])
            if explicit.get("assignment_quantile") is not None
            else None
        ),
        "feature_encoding": explicit.get("feature_encoding", "binary"),
    }
    quantile = settings["assignment_quantile"]
    if quantile is not None and not 0.0 < quantile <= 1.0:
        raise ValueError("assignment_quantile must be in (0, 1]")
    return settings


def _cluster(embeddings, settings):
    if len(embeddings) == 1:
        return np.zeros(1, dtype=np.int32)
    kwargs = {"metric": "cosine", "linkage": "average"}
    if settings["n_clusters"] is None:
        kwargs.update(n_clusters=None, distance_threshold=settings["distance_threshold"])
    else:
        kwargs.update(n_clusters=min(int(settings["n_clusters"]), len(embeddings)))
    return AgglomerativeClustering(**kwargs).fit_predict(embeddings).astype(np.int32)


def fit_agglomerative_hierarchy(embeddings):
    """Fit one complete average-linkage tree reusable for threshold and fixed-K cuts."""
    values = np.asarray(embeddings, dtype=np.float32)
    if len(values) < 2:
        return {
            "children": np.empty((0, 2), dtype=np.int64),
            "distances": np.empty(0, dtype=np.float64),
            "n_samples": int(len(values)),
        }
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.0,
        metric="cosine",
        linkage="average",
        compute_distances=True,
    ).fit(values)
    return {
        "children": np.asarray(clusterer.children_, dtype=np.int64),
        "distances": np.asarray(clusterer.distances_, dtype=np.float64),
        "n_samples": int(len(values)),
    }


def cut_agglomerative_hierarchy(
    hierarchy,
    *,
    n_clusters=None,
    distance_threshold=None,
):
    """Cut a saved sklearn hierarchy without repeating the O(n²) tree fit."""
    n_samples = int(hierarchy["n_samples"])
    if n_samples == 0:
        return np.empty(0, dtype=np.int32)
    if n_samples == 1:
        return np.zeros(1, dtype=np.int32)
    if (n_clusters is None) == (distance_threshold is None):
        raise ValueError("Specify exactly one of n_clusters or distance_threshold")
    children = np.asarray(hierarchy["children"], dtype=np.int64)
    if n_clusters is not None:
        target = min(max(int(n_clusters), 1), n_samples)
        merge_count = n_samples - target
    else:
        distances = np.asarray(hierarchy["distances"], dtype=np.float64)
        merge_count = int(np.searchsorted(
            distances,
            float(distance_threshold),
            side="right",
        ))

    active = set(range(n_samples))
    for merge_index, (left, right) in enumerate(children[:merge_count]):
        active.discard(int(left))
        active.discard(int(right))
        active.add(n_samples + merge_index)

    labels = np.empty(n_samples, dtype=np.int32)
    for label, root in enumerate(sorted(active)):
        stack = [int(root)]
        while stack:
            node = stack.pop()
            if node < n_samples:
                labels[node] = label
            else:
                left, right = children[node - n_samples]
                stack.extend((int(left), int(right)))
    return labels


def _centroid(values):
    center = values.mean(axis=0)
    return center / max(float(np.linalg.norm(center)), 1e-12)


def fit_semantic_space(config, train_records, *, embedder: Embedder = embed_texts):
    """Fit from train texts only. Labels below are post-hoc metadata."""
    space = unique_claim_space(train_records)
    occurrences = space["occurrences"]
    texts = space["texts"]
    occurrence_to_unique = space["occurrence_to_unique"]
    model_name = config.get("clustering", {}).get(
        "embedding_model", config.get("pipeline", {}).get("embedding_model", "tf-idf")
    )
    embeddings, embedding_transformer, embedding_state_signature = fit_text_embedding_space(
        texts,
        model_name,
        embedder=embedder,
    )
    settings = _settings(config)
    raw_ids = _cluster(embeddings, settings)
    return build_semantic_model_from_partition(
        config,
        occurrences=occurrences,
        texts=texts,
        occurrence_to_unique=occurrence_to_unique,
        embeddings=embeddings,
        raw_ids=raw_ids,
        embedding_transformer=embedding_transformer,
        embedding_state_signature=embedding_state_signature,
        train_records=train_records,
        embedder=embedder,
    )


def build_semantic_model_from_partition(
    config,
    *,
    occurrences,
    texts,
    occurrence_to_unique,
    embeddings,
    raw_ids,
    embedding_transformer=None,
    embedding_state_signature=None,
    train_records=None,
    embedder: Embedder = embed_texts,
):
    """Build the frozen semantic model from a label-agnostic train partition."""
    model_name = config.get("clustering", {}).get(
        "embedding_model", config.get("pipeline", {}).get("embedding_model", "tf-idf")
    )
    settings = _settings(config)
    centroids, metadata, assignment_max_distances = [], [], []
    for raw_id in sorted(set(raw_ids.tolist())):
        if int(raw_id) < 0:
            # Density-clustering noise is intentionally left unassigned.
            continue
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
        assignment_radius = (
            float(np.quantile(distances, settings["assignment_quantile"]))
            if settings["assignment_quantile"] is not None
            else float(settings["max_assign_distance"])
        )
        centroids.append(center)
        assignment_max_distances.append(assignment_radius)
        metadata.append({
            "cluster_id": cluster_id, "feature": f"cot_{cluster_id}",
            "medoid": texts[medoid_index],
            "examples": [texts[i] for i in accepted[np.argsort(distances)[:10]]],
            "unique_claims": int(len(accepted)), "occurrences": int(len(occurrence_indices)),
            "unique_clients": int(len(clients)),
            "compactness_mean_distance": float(distances.mean()),
            "assignment_distance_p95": float(np.quantile(distances, 0.95)),
            "assignment_radius": assignment_radius,
            "label_counts": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        })
    if not centroids:
        raise ValueError("No semantic clusters meet min_client_coverage")
    model = {
        "centroids": np.vstack(centroids),
        "assignment_max_distances": assignment_max_distances,
        "feature_names": [row["feature"] for row in metadata],
        "cluster_meta": metadata, "embedding_model": model_name, "settings": settings,
        "embedding_transformer": embedding_transformer,
        "embedding_state_signature": embedding_state_signature,
        "formation_signature": fingerprint({
            "texts": texts,
            "model": model_name,
            "embedding_state_signature": embedding_state_signature,
            "settings": settings,
            "partition": np.asarray(raw_ids, dtype=np.int32).tolist(),
        }),
    }
    if train_records is not None:
        model.update(fit_supervised_selection(
            config,
            transform_semantic_space(
                config,
                train_records,
                model,
                embedder=embedder,
            ),
        ))
    else:
        model.update({
            "selected_feature_names": model["feature_names"],
            "selection_scores": {},
            "selection_mode": "disabled",
        })
    return model


def transform_semantic_space(config, records, model, *, embedder: Embedder = embed_texts):
    """Assign any split to frozen train centroids without consulting labels."""
    space = unique_claim_space(records)
    occurrences = space["occurrences"]
    texts = space["texts"]
    occurrence_to_unique = space["occurrence_to_unique"]
    embeddings = transform_text_embedding_space(
        texts,
        model["embedding_model"],
        model.get("embedding_transformer"),
        embedder=embedder,
    )
    features, _ = transform_precomputed_claim_space(
        records,
        space,
        embeddings,
        model,
    )
    return features


def nearest_centroid_assignments(
    embeddings,
    centroids,
    *,
    max_distance,
    batch_size=2048,
):
    """Assign embeddings in bounded-memory batches."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    nearest = np.full(len(embeddings), -1, dtype=np.int32)
    nearest_distance = np.full(len(embeddings), np.inf, dtype=np.float32)
    for start in range(0, len(embeddings), int(batch_size)):
        stop = min(start + int(batch_size), len(embeddings))
        distances = cosine_distances(embeddings[start:stop], centroids)
        local_nearest = distances.argmin(axis=1)
        local_distance = distances[
            np.arange(stop - start),
            local_nearest,
        ]
        if np.ndim(max_distance) == 0:
            local_threshold = float(max_distance)
        else:
            thresholds = np.asarray(max_distance, dtype=np.float32)
            if len(thresholds) != len(centroids):
                raise ValueError(
                    "Per-cluster assignment thresholds do not match centroids"
                )
            local_threshold = thresholds[local_nearest]
        accepted = local_distance <= local_threshold
        accepted_indices = np.flatnonzero(accepted) + start
        nearest[accepted_indices] = local_nearest[accepted].astype(np.int32)
        nearest_distance[start:stop] = local_distance.astype(np.float32)
    return nearest, nearest_distance


def transform_precomputed_claim_space(records, space, embeddings, model):
    """Build client features and auditable occurrence assignments."""
    occurrences = space["occurrences"]
    occurrence_to_unique = np.asarray(
        space["occurrence_to_unique"],
        dtype=np.int32,
    )
    vectors = {
        canonical_entity_id(row["customer_id"]): np.zeros(
            len(model["feature_names"]), dtype=np.float32
        )
        for row in records
    }
    unique_assignments, unique_distances = nearest_centroid_assignments(
        embeddings,
        model["centroids"],
        max_distance=model.get(
            "assignment_max_distances",
            model["settings"]["max_assign_distance"],
        ),
    )
    assignment_rows = []
    for occurrence_index, row in enumerate(occurrences):
        unique_index = int(occurrence_to_unique[occurrence_index])
        cluster_index = int(unique_assignments[unique_index])
        if cluster_index >= 0:
            vectors[row["customer_id"]][cluster_index] += 1.0
        assignment_rows.append({
            "claim_id": row["claim_id"],
            "customer_id": canonical_entity_id(row["customer_id"]),
            "label": int(row.get("label", -1)),
            "normalized_text": row["normalized_text"],
            "cluster_index": cluster_index,
            "cluster_id": (
                model["cluster_meta"][cluster_index]["cluster_id"]
                if cluster_index >= 0
                else None
            ),
            "assignment_distance": float(unique_distances[unique_index]),
            "assigned": bool(cluster_index >= 0),
        })
    encoding, rows = model["settings"]["feature_encoding"], []
    for record in records:
        cid = canonical_entity_id(record["customer_id"])
        values = vectors[cid]
        if encoding == "binary":
            values = (values > 0).astype(np.float32)
        elif encoding == "normalized_count":
            values = values / max(float(values.sum()), 1.0)
        elif encoding != "raw_count":
            raise ValueError(f"Unknown feature_encoding: {encoding}")
        row = {"customer_id": cid, "label": int(record.get("label", -1))}
        row.update(dict(zip(model["feature_names"], values.astype(float))))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(assignment_rows)


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

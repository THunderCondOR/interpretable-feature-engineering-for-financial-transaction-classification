"""
src/utils/filtration.py

Utilities for filtering and cleaning atomic claims.

Functions:
    filter_russian            — keep only Cyrillic texts
    clean_text                — remove boilerplate phrases
    map_to_russian_gender     — normalize gender strings (kept for backward compat)
    select_majority_facts     — keep facts matching majority predicted label
    label_based_outlier_detection — remove claims inconsistent with neighbors
    filter_cluster_ids_by_class_diff — remove uninformative clusters
"""

import re
from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def filter_russian(texts: list[str]) -> list[str]:
    """Keep only strings that contain at least one Cyrillic character."""
    return [t for t in texts if re.search(r"[а-яА-ЯёЁ]", t)]


def clean_text(text: str) -> str:
    """Remove common boilerplate phrases from atomic claims."""
    boilerplate = [
        r"\bКлиент\b", r"\bВсе транзакции\b", r"\bTransactions\b",
        r"\btransaction[s]?\b", r"\bтранзакци[яй]\b",
    ]
    for bp in boilerplate:
        text = re.sub(bp, "", text, flags=re.IGNORECASE)
    return text.strip()


# ---------------------------------------------------------------------------
# Gender helpers (kept for backward compatibility with gender pipeline)
# ---------------------------------------------------------------------------

def map_to_russian_gender(gender_str: str) -> str | None:
    """Map any gender string to canonical 'мужчина' / 'женщина'."""
    if not gender_str:
        return None
    gender_map = {
        "male": "мужчина", "m": "мужчина", "man": "мужчина",
        "female": "женщина", "f": "женщина", "woman": "женщина",
        "м": "мужчина", "ж": "женщина",
        "м.": "мужчина", "ж.": "женщина",
        "мужчина": "мужчина", "женщина": "женщина",
    }
    return gender_map.get(gender_str.lower(), None)


def select_majority_facts(facts_with_gender: list[dict]) -> tuple:
    """
    Given a list of {fact, gender} dicts, return (majority_gender, majority_facts).
    Used in the gender pipeline to filter predictions by majority vote.
    """
    genders = [e["gender"] for e in facts_with_gender if e.get("gender")]
    if not genders:
        return None, []
    majority_gender = Counter(genders).most_common(1)[0][0]
    majority_facts  = [e["fact"] for e in facts_with_gender if e.get("gender") == majority_gender]
    return majority_gender, majority_facts


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def label_based_outlier_detection(
    embeddings: np.ndarray,
    labels: list,
    top_k: int = 5,
) -> list[bool]:
    """
    Flag a claim as an outlier if fewer than half of its top-k neighbors
    share the same label.

    Returns:
        List of bools, True = outlier (should be removed).
    """
    nn = NearestNeighbors(n_neighbors=top_k + 1, metric="cosine", n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    outlier_flags = []
    for i, idx in enumerate(indices):
        topk_labels    = [labels[j] for j in idx[1:]]  # skip self
        n_same         = sum(l == labels[i] for l in topk_labels)
        outlier_flags.append(n_same < top_k / 2)

    return outlier_flags


# ---------------------------------------------------------------------------
# Cluster filtering by class separability
# ---------------------------------------------------------------------------

def filter_cluster_ids_by_class_diff(
    cluster_ids: np.ndarray,
    labels: list,
    num_labels: int,
    min_diff: float = 0.02,
) -> np.ndarray:
    """
    Set cluster id to -1 for clusters that don't show enough class separation.

    For binary classification: keeps clusters where
        |prop(label_0) - prop(label_1)| >= min_diff

    For multi-class: keeps clusters where
        max(class_proportions) - min(class_proportions) >= min_diff

    Args:
        cluster_ids: array of cluster assignments, -1 = already noise
        labels:      list of class labels (ints), same length as cluster_ids
        num_labels:  number of distinct label classes
        min_diff:    minimum class proportion spread to keep cluster

    Returns:
        Modified cluster_ids array (noise clusters set to -1).
    """
    cluster_ids = np.array(cluster_ids, dtype=np.int32)
    labels_arr  = np.array(labels, dtype=np.int32)

    unique_clusters = set(int(c) for c in cluster_ids if c >= 0)
    bad_clusters    = set()

    for clu in unique_clusters:
        mask        = cluster_ids == clu
        clu_labels  = labels_arr[mask]
        total       = len(clu_labels)
        if total == 0:
            bad_clusters.add(clu)
            continue

        proportions = np.array([
            np.sum(clu_labels == cls) / total
            for cls in range(num_labels)
        ])

        spread = proportions.max() - proportions.min()
        if spread < min_diff:
            bad_clusters.add(clu)

    result = cluster_ids.copy()
    for clu in bad_clusters:
        result[result == clu] = -1

    return result

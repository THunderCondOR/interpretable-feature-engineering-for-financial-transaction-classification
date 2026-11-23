"""Train-only quality filters for frozen semantic-cluster features.

The filters in this module never inspect validation/test labels.  They operate
on cluster text/geometry recorded on train and on the train client-presence
matrix.  Supervised incremental selection remains a separate downstream step.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

CLEANUP_VARIANTS = ("current", "claim_clean", "cluster_clean", "full_clean")

_LOW_INFORMATION_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\b(?:overall )?(?:transaction|spending|behavioral) profile aligns with\b",
        r"\b(?:overall )?(?:transaction|spending|behavioral) pattern aligns with\b",
        r"\bsimilar to (?:that of )?typical (?:clients|groups|profiles)\b",
        r"\btypical (?:demographic|reference) groups?\b",
        r"\b(?:higher|lower)[- ]risk (?:signals?|profile)\b",
        r"\bfinancial life\b",
        r"\btransitional life stage\b",
    )
)

_EVIDENCE_TERMS = re.compile(
    r"\b(?:transaction|category|operation|amount|value|balance|activity|active|"
    r"frequency|count|share|volume|inflow|outflow|withdrawal|deposit|payment|"
    r"transfer|purchase|cash|recency|temporal|day|week|month|median|mean|"
    r"spend(?:ing)?|expense|income|credit|debit|loan)\w*\b",
    re.I,
)


def low_information_claim(text: str) -> bool:
    """Conservative deterministic rejection of boilerplate/meta claims."""
    value = " ".join(str(text or "").split())
    if not value:
        return True
    if any(pattern.search(value) for pattern in _LOW_INFORMATION_PATTERNS):
        return True
    # An interpretation with no observable transaction/account dimension is
    # not a useful atomic feature.  Evidence-bearing interpretations survive.
    speculative = re.search(r"\b(?:may|might|could|likely|appears? to)\b", value, re.I)
    return bool(speculative and not _EVIDENCE_TERMS.search(value))


def _meta_by_feature(cluster_meta: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("feature")): row for row in cluster_meta if row.get("feature")}


def _claim_quality(feature: str, metadata: dict[str, dict[str, Any]]) -> bool:
    row = metadata.get(feature, {})
    texts = [str(row.get("medoid", "")), *map(str, row.get("examples", []))]
    texts = [text for text in texts if text.strip()]
    if not texts:
        return False
    rejected = sum(low_information_claim(text) for text in texts)
    return not low_information_claim(texts[0]) and rejected / len(texts) < 0.5


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def select_clean_cluster_features(
    train: pd.DataFrame,
    cluster_meta: list[dict[str, Any]],
    *,
    variant: str,
    compactness_quantile: float = 0.90,
    redundancy_jaccard: float = 0.90,
) -> tuple[list[str], dict[str, Any]]:
    """Return a frozen train-derived feature subset and an audit report."""
    if variant not in CLEANUP_VARIANTS:
        raise ValueError(f"Unknown cleanup variant: {variant}")
    names = [column for column in train if column.startswith("cot_")]
    metadata = _meta_by_feature(cluster_meta)
    selected = list(names)
    reasons: dict[str, list[str]] = {name: [] for name in names}

    if variant in {"claim_clean", "full_clean"}:
        for name in names:
            if not _claim_quality(name, metadata):
                reasons[name].append("low_information_claim")

    geometry = variant in {"cluster_clean", "full_clean"}
    if geometry and names:
        n_clients = len(train)
        # A fixed five-client rule admits fragile near-singletons on small
        # folds.  This bounded relative rule remains usable on large datasets.
        min_clients = max(10, min(25, math.ceil(0.005 * n_clients)))
        max_clients = max(min_clients, math.floor(0.90 * n_clients))
        compact_values = np.asarray(
            [
                float(metadata.get(name, {}).get("compactness_mean_distance", np.inf))
                for name in names
            ]
        )
        finite = compact_values[np.isfinite(compact_values)]
        compact_limit = float(np.quantile(finite, compactness_quantile)) if len(finite) else np.inf
        for index, name in enumerate(names):
            row = metadata.get(name, {})
            coverage = int(row.get("assigned_unique_clients", row.get("unique_clients", 0)))
            if coverage < min_clients:
                reasons[name].append("low_train_client_coverage")
            if coverage > max_clients:
                reasons[name].append("near_universal_cluster")
            if compact_values[index] > compact_limit:
                reasons[name].append("diffuse_cluster")
            values = train[name].to_numpy() != 0
            if values.all() or not values.any():
                reasons[name].append("constant_train_feature")

        # Remove redundant client-presence columns after the cheap gates.  Keep
        # the more prevalent/compact representative deterministically.
        survivors = [name for name in names if not reasons[name]]
        survivors.sort(
            key=lambda name: (
                -int(
                    metadata.get(name, {}).get(
                        "assigned_unique_clients", metadata.get(name, {}).get("unique_clients", 0)
                    )
                ),
                float(metadata.get(name, {}).get("compactness_mean_distance", np.inf)),
                name,
            )
        )
        kept: list[str] = []
        binary = {name: train[name].to_numpy() != 0 for name in survivors}
        for name in survivors:
            duplicate = next(
                (
                    other
                    for other in kept
                    if _jaccard(binary[name], binary[other]) >= redundancy_jaccard
                ),
                None,
            )
            if duplicate is None:
                kept.append(name)
            else:
                reasons[name].append(f"redundant_with:{duplicate}")

    selected = [name for name in names if not reasons[name]]
    report = {
        "variant": variant,
        "train_only": True,
        "input_features": len(names),
        "selected_features": len(selected),
        "removed_features": len(names) - len(selected),
        "parameters": {
            "compactness_quantile": compactness_quantile,
            "redundancy_jaccard": redundancy_jaccard,
        },
        "removed": {name: value for name, value in reasons.items() if value},
    }
    return selected, report

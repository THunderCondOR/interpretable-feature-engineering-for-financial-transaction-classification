"""Build English, dataset-aware profiles for prompts and downstream models."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from interpretable_features.data.entity_ids import canonical_entity_id
from interpretable_features.data.profiles import (
    format_client_profile,
    format_legacy_mean_category_summary,
    format_robust_summary,
    robust_statistics_payload,
)


def get_summary_fn(config: dict) -> Callable:
    """Return the single supported client-profile formatter."""
    return lambda client_df, _category_label=None: format_client_profile(client_df, config)


def build_dataset_summary_str(df: pd.DataFrame, config: dict) -> str:
    """Build a label-conditioned summary from the supplied training rows only."""
    labelled = df[df["label"] >= 0] if "label" in df else df
    if config.get("statistics", {}).get("summary_profile") == "legacy_mean_categories":
        return format_legacy_mean_category_summary(labelled, config)
    return format_robust_summary(robust_statistics_payload(labelled, config))


def build_all_client_stats(df: pd.DataFrame, config: dict) -> list[dict]:
    """Render one deterministic model-facing profile per client."""
    label_names = config["dataset"]["label_names"]
    category_label = config["dataset"].get("category_label", "transaction categories")
    summary_fn = get_summary_fn(config)
    records = []
    for customer_id, client_df in df.groupby("customer_id", sort=False, observed=True):
        label = int(client_df["label"].iloc[0]) if "label" in client_df.columns else -1
        records.append(
            {
                "customer_id": canonical_entity_id(customer_id),
                "label": label,
                "label_name": label_names.get(str(label), "unknown") if label >= 0 else "unknown",
                "client_stats": summary_fn(client_df, category_label),
            }
        )
    return records

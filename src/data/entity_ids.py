"""Canonical identifiers shared by numeric and string-ID datasets."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pandas as pd


EntityId: TypeAlias = int | str


def canonical_entity_id(value: Any) -> EntityId:
    """Preserve legacy integer IDs while accepting opaque string identifiers."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Entity ID cannot be null")
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Entity ID cannot be empty")
    # CSV readers turn integer IDs into strings in some paths. Keep the old
    # integer representation so existing artifacts remain compatible.
    if text.isdecimal() or (text.startswith("-") and text[1:].isdecimal()):
        return int(text)
    return text


def canonical_entity_series(values: pd.Series) -> pd.Series:
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = list(values.cat.categories)
        if all(
            isinstance(canonical_entity_id(value), str)
            and canonical_entity_id(value) == str(value).strip()
            for value in categories
        ):
            return values
    return values.map(canonical_entity_id)


def entity_sort_key(value: Any) -> tuple[int, str]:
    canonical = canonical_entity_id(value)
    return (0 if isinstance(canonical, int) else 1, str(canonical))

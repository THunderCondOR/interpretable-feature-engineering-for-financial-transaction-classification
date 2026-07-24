"""Deterministic client-level sampling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_client_ids(
    transactions: pd.DataFrame,
    *,
    n_clients: int,
    seed: int,
) -> list[int]:
    """Sample clients by label, activity quartile, and absolute-volume quartile."""
    clients = (
        transactions.groupby("customer_id", sort=False)
        .agg(
            label=("label", "first"),
            transaction_count=("amount", "size"),
            transaction_volume=("amount", lambda values: values.abs().sum()),
        )
        .reset_index()
        .sort_values("customer_id", kind="mergesort")
        .reset_index(drop=True)
    )
    target = min(int(n_clients), len(clients))
    if target < 1:
        raise ValueError("n_clients must select at least one client")
    for column in ("transaction_count", "transaction_volume"):
        clients[f"{column}_quartile"] = pd.qcut(
            clients[column].rank(method="first"),
            4,
            labels=False,
            duplicates="drop",
        )
    clients["stratum"] = clients[
        ["label", "transaction_count_quartile", "transaction_volume_quartile"]
    ].astype(str).agg("/".join, axis=1)
    counts = clients["stratum"].value_counts().sort_index()
    exact = counts / counts.sum() * target
    allocation = np.floor(exact).astype(int)
    remainder = target - int(allocation.sum())
    for stratum in (
        (exact - allocation)
        .rename("fraction")
        .reset_index()
        .sort_values(["fraction", "stratum"], ascending=[False, True])
        .head(remainder)["stratum"]
    ):
        allocation[stratum] += 1

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for stratum, group in clients.groupby("stratum", sort=True):
        take = min(int(allocation.get(stratum, 0)), len(group))
        selected.extend(
            int(value)
            for value in rng.choice(
                group["customer_id"].to_numpy(), take, replace=False
            )
        )
    result = sorted(selected)
    if len(result) != target or len(set(result)) != target:
        raise RuntimeError(
            f"Sampling returned {len(result)} rows / {len(set(result))} "
            f"unique IDs, expected {target}"
        )
    return result

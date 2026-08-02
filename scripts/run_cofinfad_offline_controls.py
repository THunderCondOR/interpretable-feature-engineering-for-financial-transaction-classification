#!/usr/bin/env python3
"""Run COFINFAD transaction-only control and profile diagnostic ceilings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.cofinfad import DEMOGRAPHIC_COLUMNS, OPERATIONAL_COLUMNS, TARGET_ADJACENT_COLUMNS
from src.benchmarks.common import load_json
from src.experiments.artifacts import atomic_write_json, file_sha256, fingerprint


def client_table(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["tr_datetime"] = pd.to_datetime(events["tr_datetime"], errors="raise")
    grouped = events.groupby("customer_id", observed=True)
    aggregate = grouped.agg(
        tx_count=("amount", "size"), active_days=("tr_datetime", lambda x: x.dt.normalize().nunique()),
        calendar_span_days=("tr_datetime", lambda x: (x.max().normalize() - x.min().normalize()).days + 1),
        amount_mean=("amount", "mean"), amount_std=("amount", "std"),
        amount_median=("amount", "median"), amount_p95=("amount", lambda x: x.quantile(.95)),
    ).reset_index()
    type_shares = pd.crosstab(events["customer_id"], events["transaction_type"], normalize="index")
    type_shares.columns = [f"transaction_type_share_{str(value).lower()}" for value in type_shares.columns]
    aggregate = aggregate.merge(type_shares.reset_index(), on="customer_id", validate="one_to_one")
    first = grouped.first().reset_index()
    payload = [column for column in (*OPERATIONAL_COLUMNS, *DEMOGRAPHIC_COLUMNS) if column in first]
    return aggregate.merge(first[["customer_id", *payload]], on="customer_id", validate="one_to_one")


def model_frame(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    values = frame[columns].copy()
    categorical = []
    for column in values:
        if not pd.api.types.is_numeric_dtype(values[column]):
            values[column] = values[column].astype("string").fillna("not available").astype(str)
            categorical.append(column)
        else:
            values[column] = pd.to_numeric(values[column], errors="coerce").fillna(values[column].median()).fillna(0.0)
    return values, categorical


def scores(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** .5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/isolated/baselines/cofinfad/offline_controls.json"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    plan = {
        "mode": "execute" if args.execute else "dry-run", "dataset": manifest.get("dataset"),
        "views": ["transaction_only_negative_control", "operational_primary", "full_profile_diagnostic_ceiling"],
        "teacher": "published_churn_probability", "model": "CatBoostRegressor",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    from catboost import CatBoostRegressor

    events = pd.read_parquet(manifest["events"])
    customers = client_table(events)
    teacher = pd.read_csv(manifest["teacher_scores"], dtype={"customer_id": str})
    customers["customer_id"] = customers["customer_id"].astype(str)
    customers = customers.merge(teacher, on="customer_id", validate="one_to_one")
    transaction_columns = [
        column for column in customers
        if column.startswith(("tx_", "calendar_", "amount_", "transaction_type_"))
        or column == "active_days"
    ]
    operational = [column for column in OPERATIONAL_COLUMNS if column in customers]
    demographics = [column for column in DEMOGRAPHIC_COLUMNS if column in customers]
    unique = lambda values: list(dict.fromkeys(values))
    views = {
        "transaction_only_negative_control": transaction_columns,
        "operational_primary": unique([*transaction_columns, *operational]),
        "full_profile_diagnostic_ceiling": unique([*transaction_columns, *operational, *demographics]),
    }
    forbidden = set(TARGET_ADJACENT_COLUMNS) | {"churn_probability", "label"}
    if any(forbidden & set(columns) for columns in views.values()):
        raise RuntimeError("Target-adjacent feature leakage")
    ids = {
        split: set(json.loads(Path(manifest["roles"][split]).read_text()))
        for split in ("train", "val", "test")
    }
    results = []
    for view, columns in views.items():
        values, categorical = model_frame(customers, columns)
        masks = {split: customers["customer_id"].isin(split_ids) for split, split_ids in ids.items()}
        model = CatBoostRegressor(
            iterations=600, depth=7, learning_rate=.05, loss_function="RMSE",
            random_seed=args.seed, verbose=False, thread_count=-1,
        )
        model.fit(
            values.loc[masks["train"]], customers.loc[masks["train"], "churn_probability"],
            cat_features=categorical,
            eval_set=(values.loc[masks["val"]], customers.loc[masks["val"], "churn_probability"]),
            early_stopping_rounds=75,
        )
        for split in ("val", "test"):
            results.append({
                "view": view, "split": split, "n_features": len(columns),
                **scores(
                    customers.loc[masks[split], "churn_probability"],
                    np.clip(model.predict(values.loc[masks[split]]), 0, 1),
                ),
            })
    payload = {
        **plan, "status": "completed", "manifest_signature": manifest["manifest_signature"],
        "events_sha256": file_sha256(manifest["events"]), "results": results,
        "interpretation": {
            "transaction_only_negative_control": "Tests whether transactions alone carry the published score.",
            "operational_primary": "Matches the evidence available to the LLM.",
            "full_profile_diagnostic_ceiling": "Includes demographics only as an offline ceiling; never enters prompts.",
        },
    }
    payload["result_signature"] = fingerprint(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

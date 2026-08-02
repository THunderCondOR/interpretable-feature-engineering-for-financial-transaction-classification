"""Prepare the COFINFAD operational natural-fidelity benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.benchmarks.common import (
    download_verified,
    id_hash,
    sha256,
    stable_ids,
    write_ids,
)
from src.experiments.artifacts import atomic_write_json, fingerprint


NAME = "cofinfad_operational_fidelity"
PROTOCOL = "score_activity_stratified_7500_seed137"
REVISION = "f7b6a9f45bd75fba9f791238dee85c5028ef5b19"
SEED = 137
SAMPLE_SIZE = 7_500
EXPECTED_CLIENTS = 48_723
EXPECTED_TRANSACTIONS = 3_159_157

FILES = {
    "customer_data.csv": (
        f"https://huggingface.co/datasets/luisdavidtrejosrojas/cofinfad/resolve/{REVISION}/customer_data.csv?download=true",
        "bb3865b68c247caaa28821238c3d9fa9c745ca8837c42f89b5d7e310beb98c8d",
    ),
    "transactions_data.csv": (
        f"https://huggingface.co/datasets/luisdavidtrejosrojas/cofinfad/resolve/{REVISION}/transactions_data.csv?download=true",
        "09fa21b8d74692cbfbf10ee58b55b00c874280a57aa926650e2fe55c60859ec6",
    ),
}

DEMOGRAPHIC_COLUMNS = (
    "age", "gender", "location", "income_bracket", "occupation",
    "education_level", "marital_status", "household_size",
    "acquisition_channel",
)
TARGET_ADJACENT_COLUMNS = (
    "churn_probability", "customer_lifetime_value", "customer_segment",
    "clv_segment",
)
OPERATIONAL_COLUMNS = (
    "savings_account", "credit_card", "personal_loan", "investment_account",
    "insurance_product", "active_products", "app_logins_frequency",
    "feature_usage_diversity", "bill_payment_user", "auto_savings_enabled",
    "credit_utilization_ratio", "international_transactions",
    "failed_transactions", "base_satisfaction", "tx_satisfaction",
    "product_satisfaction", "satisfaction_score", "nps_score",
    "last_survey_date", "support_tickets_count", "resolved_tickets_ratio",
    "app_store_rating", "feedback_sentiment", "feature_requests",
    "complaint_topics",
)


def acquire(raw_root: Path) -> dict[str, Path]:
    return {
        name: download_verified(url, raw_root / name, expected)
        for name, (url, expected) in FILES.items()
    }


def _strata(customers: pd.DataFrame) -> pd.Series:
    score_decile = pd.qcut(
        customers["churn_probability"], 10, labels=False, duplicates="drop"
    )
    activity_quartile = pd.qcut(
        np.log1p(customers["tx_count"]), 4, labels=False, duplicates="drop"
    )
    return score_decile.astype(str) + "_" + activity_quartile.astype(str)


def _risk_labels(scores: pd.Series, boundaries: list[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(boundaries, dtype=float), scores, side="right")


def prepare(raw_root: Path, output_root: Path) -> Path:
    paths = acquire(raw_root)
    customers = pd.read_csv(paths["customer_data.csv"])
    transactions = pd.read_csv(paths["transactions_data.csv"])
    if len(customers) != EXPECTED_CLIENTS or customers["customer_id"].nunique() != EXPECTED_CLIENTS:
        raise ValueError("Unexpected COFINFAD customer population")
    if len(transactions) != EXPECTED_TRANSACTIONS:
        raise ValueError("Unexpected COFINFAD transaction count")
    required_tx = {"customer_id", "date", "amount", "type"}
    if set(transactions) != required_tx:
        raise ValueError(f"Unexpected COFINFAD transaction schema: {set(transactions)}")
    if set(transactions["customer_id"].unique()) != set(customers["customer_id"]):
        raise ValueError("COFINFAD customer and transaction populations differ")
    if transactions[list(required_tx)].isna().any().any():
        raise ValueError("COFINFAD raw transactions unexpectedly contain missing values")
    if (transactions["amount"] <= 0).any():
        raise ValueError("COFINFAD transaction amounts must be unsigned positive COP values")
    expected_types = {"Transfer", "Payment", "Withdrawal", "Deposit"}
    if set(transactions["type"]) != expected_types:
        raise ValueError("Unexpected COFINFAD transaction types")
    if customers["churn_probability"].isna().any():
        raise ValueError("Published churn scores must be complete")

    customers = customers.copy()
    customers["_stratum"] = _strata(customers)
    sampled, _ = train_test_split(
        customers,
        train_size=SAMPLE_SIZE,
        random_state=SEED,
        stratify=customers["_stratum"],
    )
    sampled = sampled.copy()
    sampled["_score_decile"] = pd.qcut(
        sampled["churn_probability"], 10, labels=False, duplicates="drop"
    )
    train, remainder = train_test_split(
        sampled,
        test_size=0.40,
        random_state=SEED,
        stratify=sampled["_score_decile"],
    )
    val, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=SEED,
        stratify=remainder["_score_decile"],
    )
    boundaries = [
        float(value)
        for value in train["churn_probability"].quantile([0.25, 0.50, 0.75])
    ]
    sampled["label"] = _risk_labels(sampled["churn_probability"], boundaries)
    role_frames = {"train": train, "val": val, "test": test}
    role_ids = {
        role: stable_ids(frame["customer_id"])
        for role, frame in role_frames.items()
    }
    pilot_population = sampled[sampled["customer_id"].isin(val["customer_id"])].copy()
    pilot_population["label"] = _risk_labels(
        pilot_population["churn_probability"], boundaries
    )
    pilot_population["_activity_quartile"] = pd.qcut(
        np.log1p(pilot_population["tx_count"]), 4, labels=False, duplicates="drop"
    )
    pilot_population["_pilot_stratum"] = (
        pilot_population["label"].astype(str)
        + "_" + pilot_population["_activity_quartile"].astype(str)
    )
    pilot, _ = train_test_split(
        pilot_population,
        train_size=400,
        random_state=SEED,
        stratify=pilot_population["_pilot_stratum"],
    )
    role_ids["pilot_val"] = stable_ids(pilot["customer_id"])

    sampled_ids_numeric = set(sampled["customer_id"].astype(int))
    sampled_transactions = transactions[
        transactions["customer_id"].isin(sampled_ids_numeric)
    ].copy()
    customer_payload = sampled.drop(
        columns=["_stratum", "_score_decile"], errors="ignore"
    )
    sampled_transactions = sampled_transactions.merge(
        customer_payload,
        on="customer_id",
        how="left",
        validate="many_to_one",
    ).rename(columns={
        "date": "tr_datetime",
        "type": "mcc_code_desc",
    })
    sampled_transactions["customer_id"] = sampled_transactions["customer_id"].astype(str)
    sampled_transactions["transaction_type"] = sampled_transactions["mcc_code_desc"]

    protocol_root = output_root / NAME / PROTOCOL
    protocol_root.mkdir(parents=True, exist_ok=True)
    events_path = protocol_root / "events.parquet"
    sampled_transactions.to_parquet(events_path, index=False, compression="zstd")
    score_path = protocol_root / "teacher_scores.csv"
    customer_payload[["customer_id", "label", "churn_probability"]].sort_values(
        "customer_id"
    ).to_csv(score_path, index=False)
    id_paths = {
        role: write_ids(protocol_root / f"{role}_ids.json", ids)
        for role, ids in role_ids.items()
    }
    label_by_id = dict(
        zip(customer_payload["customer_id"].astype(str), customer_payload["label"].astype(int))
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": NAME,
        "protocol": PROTOCOL,
        "revision": REVISION,
        "seed": SEED,
        "population_clients": EXPECTED_CLIENTS,
        "population_transactions": EXPECTED_TRANSACTIONS,
        "sample_clients": SAMPLE_SIZE,
        "sample_transactions": int(len(sampled_transactions)),
        "raw_exact_duplicate_transaction_rows": int(transactions.duplicated().sum()),
        "duplicate_policy": "preserved_as_published",
        "events": str(events_path),
        "teacher_scores": str(score_path),
        "raw_sha256": {name: sha256(path) for name, path in paths.items()},
        "events_sha256": sha256(events_path),
        "roles": {role: str(path) for role, path in id_paths.items()},
        "counts": {role: len(ids) for role, ids in role_ids.items()},
        "id_hashes": {role: id_hash(ids) for role, ids in role_ids.items()},
        "risk_quartile_boundaries_train_only": boundaries,
        "class_counts": {
            role: {
                str(label): sum(label_by_id[value] == label for value in ids)
                for label in range(4)
            }
            for role, ids in role_ids.items()
        },
        "prompt_profile": {
            "included_operational_columns": list(OPERATIONAL_COLUMNS),
            "excluded_demographic_columns": list(DEMOGRAPHIC_COLUMNS),
            "excluded_target_adjacent_columns": list(TARGET_ADJACENT_COLUMNS),
        },
    }
    manifest["manifest_signature"] = fingerprint(manifest)
    manifest_path = protocol_root / "benchmark_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path

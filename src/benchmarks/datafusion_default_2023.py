"""Prepare Data Fusion Contest 2023 default as an isolated fixed-split benchmark."""

from __future__ import annotations

import json
import pickle
import zipfile
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


NAME = "datafusion_default_2023"
PROTOCOL = "stratified_60_20_20_seed137"
SEED = 137
EXPECTED_CLIENTS = 7_080
EXPECTED_TRANSACTIONS = 2_124_000
EXPECTED_POSITIVES = 262
EXPECTED_TRANSACTIONS_PER_CLIENT = 300

FILES = {
    "transactions.zip": (
        "https://storage.yandexcloud.net/ds-ods/files/materials/c7b69754/transactions.zip",
        "6b4cab21d07d12c1ce0ad6b66d4baf1125ab1d330984dddbef7e750823530266",
    ),
    "train_target.zip": (
        "https://storage.yandexcloud.net/ds-ods/files/materials/a4faa80b/train_target.zip",
        "cb72d692155a8edf09f65050e58a667a34382c3e681864198e91f9f22a9afa33",
    ),
    "model.zip": (
        "https://storage.yandexcloud.net/ds-ods/files/materials/750fd067/model.zip",
        "401e01977945c65a473439caa95809e4687d59c82666899fe5d1609b827e67b9",
    ),
    "mcc_codes.csv": (
        "https://storage.yandexcloud.net/ds-ods/files/materials/a93abd22/mcc_codes.csv",
        "21949a20cca615ae5069a94b01aafb2900f845032e843fb72fd6f1d043d1ce94",
    ),
    "currency_rk.csv": (
        "https://storage.yandexcloud.net/ds-ods/files/materials/420d4b9c/currency_rk.csv",
        "7925b9f27b4605b5f25b479cb9629555dd437eeaf51bd07254ce1f080cf5b65b",
    ),
    "visa_mcc.csv": (
        "https://gist.githubusercontent.com/GeoffreyFrogeye/9361b4dc998812170c03e406ba2640e2/raw/MCC.csv",
        "29896ce0c43312927d79d1ba22027c97a6594cab886d53cb46e69a51b598450c",
    ),
}

# Codes present in the contest data but absent from the pinned Visa table.
# Titles are conservative English translations of the contest's own MCC table.
MCC_SUPPLEMENT = {
    3068: "Air Astana",
    3211: "Airline merchant",
    3245: "EasyJet",
    3301: "Wizz Air",
    3692: "DoubleTree Hotels",
    4011: "Railroads — freight transport",
    4816: "Computer network and information services",
    5192: "Books, periodicals, and newspapers",
    5599: "Miscellaneous vehicle, aircraft, and farm-equipment dealers",
    5815: "Digital goods — audiovisual media, books, movies, and music",
    5816: "Digital goods — games",
    5817: "Digital goods — applications excluding games",
    5818: "Digital goods — multi-category",
    5932: "Antique shops — sales, repair, and restoration",
    5965: "Direct marketing — combination catalog and retail merchants",
    6050: "Quasi-cash — financial institutions",
    6399: "Insurance — not elsewhere classified",
    6513: "Real-estate agents and managers — rentals",
    6532: "Payment transaction — financial institution",
    6536: "Domestic card-to-card money transfer — credit",
    6537: "Cross-border card-to-card money transfer — credit",
    6538: "Card-to-card money transfer — debit",
    6540: "Non-financial institution stored-value account funding",
    6555: "Money transfer",
    7349: "Building cleaning and maintenance services",
    7832: "Motion-picture theaters",
    8071: "Dental and medical laboratories",
}


def acquire(raw_root: Path) -> dict[str, Path]:
    return {
        name: download_verified(url, raw_root / name, expected)
        for name, (url, expected) in FILES.items()
    }


def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {members}")
        with archive.open(members[0]) as stream:
            return pd.read_csv(stream)


def english_mcc_mapping(visa_path: Path) -> dict[int, str]:
    frame = pd.read_csv(visa_path)
    frame["MCC"] = pd.to_numeric(frame["MCC"], errors="coerce")
    frame = frame.dropna(subset=["MCC", "MERCHANT TYPE"])
    mapping = {
        int(code): " / ".join(dict.fromkeys(group["MERCHANT TYPE"].astype(str)))
        for code, group in frame.groupby("MCC", sort=True)
    }
    mapping.update(MCC_SUPPLEMENT)
    mapping[-1] = "Unknown merchant category"
    return mapping


def _split(labels: pd.DataFrame) -> dict[str, list[str]]:
    train_val, test = train_test_split(
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels["label"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=SEED,
        stratify=train_val["label"],
    )
    return {
        "train": stable_ids(train["customer_id"]),
        "val": stable_ids(val["customer_id"]),
        "test": stable_ids(test["customer_id"]),
    }


def _pilot_ids(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    validation_ids: list[str],
    size: int = 400,
) -> list[str]:
    validation = labels[labels["customer_id"].isin(validation_ids)].copy()
    positives = validation.loc[validation["label"] == 1, "customer_id"].tolist()
    if len(positives) >= size:
        raise ValueError("Pilot size cannot accommodate every positive validation client")
    volume = (
        events[events["customer_id"].isin(validation_ids)]
        .assign(_abs=lambda frame: frame["amount"].abs())
        .groupby("customer_id", observed=True)["_abs"]
        .sum()
    )
    negatives = validation[validation["label"] == 0].copy()
    negatives["volume"] = negatives["customer_id"].map(volume)
    negatives["volume_quartile"] = pd.qcut(
        negatives["volume"], 4, labels=False, duplicates="drop"
    )
    selected = []
    remaining = size - len(positives)
    counts = negatives["volume_quartile"].value_counts(normalize=True).sort_index()
    allocation = (counts * remaining).astype(int)
    for quartile in counts.index:
        take = int(allocation.loc[quartile])
        group = negatives[negatives["volume_quartile"] == quartile]
        selected.extend(group.sample(n=take, random_state=SEED + int(quartile))["customer_id"])
    missing = remaining - len(selected)
    if missing:
        available = negatives[~negatives["customer_id"].isin(selected)]
        selected.extend(available.sample(n=missing, random_state=SEED + 99)["customer_id"])
    return stable_ids([*positives, *selected])


def prepare(raw_root: Path, output_root: Path) -> Path:
    paths = acquire(raw_root)
    transactions = _read_zip_csv(paths["transactions.zip"])
    target = _read_zip_csv(paths["train_target.zip"])
    required_transactions = {
        "user_id", "mcc_code", "currency_rk", "transaction_amt", "transaction_dttm"
    }
    if set(transactions) != required_transactions or set(target) != {"user_id", "target"}:
        raise ValueError("Unexpected Data Fusion 2023 schema")
    if len(transactions) != EXPECTED_TRANSACTIONS:
        raise ValueError(f"Unexpected transaction count: {len(transactions)}")
    if target["user_id"].nunique() != EXPECTED_CLIENTS or target["target"].sum() != EXPECTED_POSITIVES:
        raise ValueError("Unexpected Data Fusion 2023 target population")
    per_client = transactions.groupby("user_id", observed=True).size()
    if set(per_client.index) != set(target["user_id"]) or not per_client.eq(EXPECTED_TRANSACTIONS_PER_CLIENT).all():
        raise ValueError("Every target client must have exactly 300 transactions")

    mcc = english_mcc_mapping(paths["visa_mcc.csv"])
    missing_mcc = sorted(set(transactions["mcc_code"].astype(int)) - set(mcc))
    if missing_mcc:
        raise ValueError(f"English MCC mapping is incomplete: {missing_mcc}")
    currencies = pd.read_csv(paths["currency_rk.csv"])
    currency_map = dict(zip(currencies["currency_rk"], currencies["Name"]))
    missing_currency = sorted(set(transactions["currency_rk"]) - set(currency_map))
    if missing_currency:
        raise ValueError(f"Currency mapping is incomplete: {missing_currency}")

    events = transactions.rename(columns={
        "user_id": "customer_id",
        "transaction_dttm": "tr_datetime",
        "transaction_amt": "amount",
    })
    events["customer_id"] = events["customer_id"].astype(str)
    events["mcc_code_desc"] = events["mcc_code"].astype(int).map(mcc)
    events["currency_name"] = events["currency_rk"].map(currency_map)
    labels = target.rename(columns={"user_id": "customer_id", "target": "label"})
    labels["customer_id"] = labels["customer_id"].astype(str)
    events = events.merge(labels, on="customer_id", how="left", validate="many_to_one")
    split_ids = _split(labels)
    pilot_ids = _pilot_ids(events, labels, split_ids["val"])

    protocol_root = output_root / NAME / PROTOCOL
    protocol_root.mkdir(parents=True, exist_ok=True)
    events_path = protocol_root / "events.parquet"
    events.to_parquet(events_path, index=False, compression="zstd")
    labels_path = protocol_root / "labels.csv"
    labels.sort_values("customer_id").to_csv(labels_path, index=False)
    roles = {**split_ids, "pilot_val": pilot_ids}
    id_paths = {role: write_ids(protocol_root / f"{role}_ids.json", ids) for role, ids in roles.items()}
    label_by_id = dict(zip(labels["customer_id"], labels["label"].astype(int)))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": NAME,
        "protocol": PROTOCOL,
        "seed": SEED,
        "events": str(events_path),
        "labels": str(labels_path),
        "raw_sha256": {name: sha256(path) for name, path in paths.items()},
        "events_sha256": sha256(events_path),
        "roles": {role: str(path) for role, path in id_paths.items()},
        "counts": {role: len(ids) for role, ids in roles.items()},
        "id_hashes": {role: id_hash(ids) for role, ids in roles.items()},
        "class_counts": {
            role: {
                str(label): sum(label_by_id[value] == label for value in ids)
                for label in (0, 1)
            }
            for role, ids in roles.items()
        },
        "mcc_mapping": {
            "source": FILES["visa_mcc.csv"][0],
            "source_sha256": FILES["visa_mcc.csv"][1],
            "observed_codes": int(events["mcc_code"].nunique()),
            "unmapped_observed_codes": [],
        },
        "amount_semantics": "signed direction in native transaction currency; currencies are never summed",
    }
    manifest["manifest_signature"] = fingerprint(manifest)
    manifest_path = protocol_root / "benchmark_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path

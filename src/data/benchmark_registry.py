"""Registry and preparation helpers for the fold-based v5 benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from src.data.mcc import MCC_TO_DESC_EN
from src.experiments.artifacts import atomic_write_json, file_sha256, fingerprint


BERKA_SEEDS = (9, 17, 101, 137, 947)
DATA_FUSION_FOLDS = 5
DATA_FUSION_SEED = 42


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    protocol: str
    primary_metric: str
    direct_selection_metric: str
    entity_column: str
    n_entities: int
    n_folds: int = 5
    modality: str = "transactions_only"
    comparison: str = "protocol_matched"


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "datafusion_education": BenchmarkSpec(
        name="datafusion_education",
        protocol="mbd_5fold_seed42",
        primary_metric="roc_auc",
        direct_selection_metric="balanced_accuracy",
        entity_column="user_id",
        n_entities=8_509,
    ),
    "berka": BenchmarkSpec(
        name="berka",
        protocol="unittab_70_30_5seed",
        primary_metric="positive_f1",
        direct_selection_metric="positive_f1",
        entity_column="loan_id",
        n_entities=682,
        comparison="protocol_matched_not_id_identical",
    ),
}


DATA_FUSION_BASELINES = {
    "Aggregation": {"mean": 0.793, "sd": 0.013, "metric": "roc_auc"},
    "CoLES": {"mean": 0.784, "sd": 0.012, "metric": "roc_auc"},
    "TabBERT": {"mean": 0.762, "sd": 0.014, "metric": "roc_auc"},
    "TabGPT": {"mean": 0.766, "sd": 0.013, "metric": "roc_auc"},
    "Supervised RNN": {"mean": 0.712, "sd": 0.015, "metric": "roc_auc"},
}

BERKA_BASELINES = {
    "UniTTab": {"mean": 0.673, "sd": 0.038, "metric": "positive_f1"},
    "TabBERT": {"mean": 0.620, "sd": 0.024, "metric": "positive_f1"},
    "LUNA": {"mean": 0.637, "sd": 0.043, "metric": "positive_f1"},
    "XGBoost": {"mean": 0.608, "sd": 0.079, "metric": "positive_f1"},
    "CatBoost": {"mean": 0.527, "sd": 0.065, "metric": "positive_f1"},
    "VAR": {"mean": 0.474, "sd": 0.007, "metric": "positive_f1"},
}


def _stable_ids(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values})


def _id_hash(values: Iterable[Any]) -> str:
    return fingerprint(_stable_ids(values))


def _write_ids(path: Path, values: Iterable[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, _stable_ids(values))
    return path


def _inner_split(
    labels: pd.DataFrame,
    outer_train_ids: Iterable[Any],
    *,
    validation_size: int,
    seed: int = 137,
) -> tuple[list[str], list[str]]:
    population = labels[
        labels["customer_id"].astype(str).isin(_stable_ids(outer_train_ids))
    ].copy()
    if validation_size >= len(population):
        raise ValueError("Inner-validation size must be smaller than outer train")
    inner_train, inner_validation = train_test_split(
        population,
        test_size=int(validation_size),
        random_state=int(seed),
        stratify=population["label"],
    )
    return (
        _stable_ids(inner_train["customer_id"]),
        _stable_ids(inner_validation["customer_id"]),
    )


def _fold_payload(
    *,
    spec: BenchmarkSpec,
    fold: int,
    outer_train: Iterable[Any],
    outer_test: Iterable[Any],
    inner_train: Iterable[Any],
    inner_validation: Iterable[Any],
    labels: pd.DataFrame,
    split_seed: int,
    raw_hashes: dict[str, str],
    split_backend: str,
) -> dict[str, Any]:
    ids = {
        "outer_train": _stable_ids(outer_train),
        "outer_test": _stable_ids(outer_test),
        "inner_train": _stable_ids(inner_train),
        "inner_validation": _stable_ids(inner_validation),
    }
    if set(ids["outer_train"]) & set(ids["outer_test"]):
        raise ValueError(f"Outer train/test overlap in fold {fold}")
    if set(ids["inner_train"]) & set(ids["inner_validation"]):
        raise ValueError(f"Inner train/validation overlap in fold {fold}")
    if set(ids["inner_train"]) | set(ids["inner_validation"]) != set(
        ids["outer_train"]
    ):
        raise ValueError(f"Inner partition does not cover outer train in fold {fold}")
    label_by_id = dict(
        zip(labels["customer_id"].astype(str), labels["label"].astype(int))
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": spec.name,
        "protocol": spec.protocol,
        "fold": int(fold),
        "split_seed": int(split_seed),
        "split_backend": split_backend,
        "primary_metric": spec.primary_metric,
        "comparison": spec.comparison,
        "modality": spec.modality,
        "raw_file_sha256": raw_hashes,
        "ids": ids,
        "id_hashes": {name: _id_hash(values) for name, values in ids.items()},
        "counts": {name: len(values) for name, values in ids.items()},
        "class_counts": {
            name: {
                str(label): sum(label_by_id[value] == label for value in values)
                for label in sorted(set(label_by_id.values()))
            }
            for name, values in ids.items()
        },
    }
    payload["fold_signature"] = fingerprint(payload)
    return payload


BERKA_TYPE = {
    "PRIJEM": "credit",
    "VYDAJ": "debit",
    "VYBER": "debit",
}
BERKA_OPERATION = {
    "VKLAD": "cash deposit",
    "PREVOD Z UCTU": "incoming bank transfer",
    "VYBER": "cash withdrawal",
    "PREVOD NA UCET": "outgoing bank transfer",
    "VYBER KARTOU": "card cash withdrawal",
}
BERKA_PURPOSE = {
    "POJISTNE": "insurance payment",
    "SLUZBY": "bank service fee",
    "UROK": "interest credited",
    "SANKC. UROK": "penalty interest",
    "SIPO": "household payment",
    "DUCHOD": "pension payment",
}


def normalize_berka(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    loan_path, transaction_path = raw_root / "loan.csv", raw_root / "trans.csv"
    for path in (loan_path, transaction_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    loans = pd.read_csv(loan_path)
    transactions = pd.read_csv(transaction_path)
    required_loan = {"loan_id", "account_id", "date", "status"}
    required_transaction = {
        "account_id", "date", "type", "operation", "amount", "balance", "k_symbol"
    }
    if required_loan - set(loans) or required_transaction - set(transactions):
        raise ValueError("Unexpected Berka raw schema")
    if set(loans["status"].dropna().unique()) != {"A", "B", "C", "D"}:
        raise ValueError("Unexpected Berka loan statuses")
    loans["label"] = loans["status"].isin(["B", "D"]).astype(int)
    loans["date"] = pd.to_datetime(loans["date"], errors="raise")
    transactions["date"] = pd.to_datetime(transactions["date"], errors="raise")
    # Loan-payment records are outcome-adjacent and never enter the benchmark.
    transactions = transactions[
        transactions["k_symbol"].fillna("").str.strip() != "UVER"
    ].copy()
    joined = loans[["loan_id", "account_id", "date", "label"]].merge(
        transactions,
        on="account_id",
        how="inner",
        suffixes=("_loan", "_transaction"),
        validate="one_to_many",
    )
    joined = joined[joined["date_transaction"] < joined["date_loan"]].copy()
    if joined["loan_id"].nunique() != len(loans):
        raise ValueError("Every Berka loan must have pre-origination transactions")
    joined["operation_type"] = (
        joined["type"].map(BERKA_TYPE).fillna(joined["type"].astype(str))
    )
    joined["operation_name"] = (
        joined["operation"].map(BERKA_OPERATION).fillna("other operation")
    )
    joined["purpose_name"] = (
        joined["k_symbol"].fillna("").str.strip().map(BERKA_PURPOSE)
        .fillna("unspecified purpose")
    )
    joined["mcc_code_desc"] = (
        joined["operation_name"] + " [" + joined["purpose_name"] + "]"
    )
    magnitude = pd.to_numeric(joined["amount"], errors="raise").abs()
    joined["amount"] = np.where(
        joined["operation_type"].eq("credit"), magnitude, -magnitude
    )
    events = joined.rename(
        columns={
            "loan_id": "customer_id",
            "date_transaction": "tr_datetime",
        }
    )[
        [
            "customer_id", "account_id", "tr_datetime", "amount",
            "balance", "operation_type", "operation_name", "purpose_name",
            "mcc_code_desc", "label",
        ]
    ].copy()
    events["customer_id"] = events["customer_id"].astype(str)
    labels = loans.rename(columns={"loan_id": "customer_id"})[
        ["customer_id", "label"]
    ].copy()
    labels["customer_id"] = labels["customer_id"].astype(str)
    hashes = {
        "loan.csv": file_sha256(loan_path),
        "trans.csv": file_sha256(transaction_path),
    }
    return events, labels, hashes


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str:
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"None of {candidates!r} found in columns {sorted(lookup)}")


def normalize_datafusion(
    raw_root: Path,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Stream the 2022 transaction CSV into one canonical parquet file."""
    transaction_candidates = (
        raw_root / "transactions.csv",
        raw_root / "transaction.csv",
        raw_root / "transactions" / "transactions.csv",
    )
    label_candidates = (raw_root / "train.csv", raw_root / "labels.csv")
    transaction_path = next((p for p in transaction_candidates if p.is_file()), None)
    label_path = next((p for p in label_candidates if p.is_file()), None)
    if transaction_path is None:
        transaction_path = next(
            iter(sorted(raw_root.rglob("transactions.csv"))), None
        )
    if label_path is None:
        label_path = next(iter(sorted(raw_root.rglob("train.csv"))), None)
    if transaction_path is None or label_path is None:
        raise FileNotFoundError(
            "Expected Data Fusion 2022 transactions.csv and train.csv under "
            f"{raw_root}"
        )
    labels_raw = pd.read_csv(label_path)
    id_column = _find_column(labels_raw, ("user_id", "client_id", "bank"))
    label_column = _find_column(
        labels_raw, ("higher_education", "target", "label")
    )
    labels = labels_raw[[id_column, label_column]].rename(
        columns={id_column: "customer_id", label_column: "label"}
    )
    labels["customer_id"] = labels["customer_id"].astype(str)
    labels["label"] = labels["label"].astype(int)
    if not labels["customer_id"].is_unique:
        raise ValueError("Data Fusion labels contain duplicate user IDs")

    reader = pacsv.open_csv(
        transaction_path,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
    )
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_map = dict(zip(labels["customer_id"], labels["label"]))
    currency_path = raw_root / "currency_rk.csv"
    currency_map: dict[str, str] = {}
    if currency_path.is_file():
        currency_frame = pd.read_csv(currency_path)
        currency_id = _find_column(currency_frame, ("currency_rk", "currency"))
        currency_name = _find_column(currency_frame, ("Name", "name"))
        currency_map = {
            str(key): str(value)
            for key, value in zip(
                currency_frame[currency_id], currency_frame[currency_name]
            )
        }
    try:
        for batch in reader:
            frame = batch.to_pandas()
            user = _find_column(frame, ("user_id", "client_id"))
            timestamp = _find_column(
                frame, ("transaction_dttm", "transaction_datetime", "datetime")
            )
            amount = _find_column(frame, ("transaction_amt", "amount"))
            mcc = _find_column(frame, ("mcc_code", "mcc"))
            currency = _find_column(frame, ("currency_rk", "currency"))
            frame["customer_id"] = frame[user].astype(str)
            frame["label"] = frame["customer_id"].map(label_map)
            frame = frame[frame["label"].notna()].copy()
            frame["tr_datetime"] = pd.to_datetime(frame[timestamp], errors="raise")
            frame["amount"] = pd.to_numeric(frame[amount], errors="raise")
            frame["mcc_code"] = pd.to_numeric(frame[mcc], errors="coerce").astype("Int64")
            frame["currency_code"] = frame[currency].astype(str)
            frame["currency_name"] = (
                frame["currency_code"].map(currency_map)
                .fillna("Currency code " + frame["currency_code"])
            )
            frame["mcc_code_desc"] = frame["mcc_code"].map(
                lambda value: MCC_TO_DESC_EN.get(
                    int(value), f"MCC {int(value)}"
                ) if pd.notna(value) else "Unknown MCC"
            )
            table = pa.Table.from_pandas(
                frame[
                    [
                        "customer_id", "tr_datetime", "amount", "mcc_code_desc",
                        "mcc_code", "currency_code", "currency_name", "label",
                    ]
                ],
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Data Fusion transaction file is empty")
    temporary_path.replace(output_path)
    observed = set(
        pd.read_parquet(output_path, columns=["customer_id"])["customer_id"].unique()
    )
    missing = set(labels["customer_id"]) - observed
    if missing:
        raise ValueError(f"{len(missing)} labeled users have no transactions")
    hashes = {
        transaction_path.name: file_sha256(transaction_path),
        label_path.name: file_sha256(label_path),
    }
    if currency_path.is_file():
        hashes[currency_path.name] = file_sha256(currency_path)
    return labels, hashes


def datafusion_folds(
    labels: pd.DataFrame,
    *,
    backend: str,
) -> list[tuple[list[str], list[str]]]:
    """Create the five MBD folds, requiring Spark for the reference protocol."""
    ordered = labels.sort_values("customer_id").reset_index(drop=True)
    if backend == "pyspark":
        try:
            import pyspark
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise RuntimeError(
                "Exact mbd_5fold_seed42 preparation requires pyspark==3.3.3. "
                "Install it or explicitly use --split-backend sklearn_approx."
            ) from exc
        if str(pyspark.__version__) != "3.3.3":
            raise RuntimeError(
                "The reference Data Fusion preparation is pinned to "
                f"pyspark==3.3.3, found {pyspark.__version__}"
            )
        spark = SparkSession.builder.master("local[*]").appName(
            "datafusion-folds"
        ).getOrCreate()
        try:
            sdf = spark.createDataFrame(
                ordered[["customer_id", "label"]].astype(
                    {"customer_id": str, "label": int}
                )
            ).orderBy("customer_id").coalesce(1)
            parts = sdf.randomSplit([0.2] * 5, seed=DATA_FUSION_SEED)
            test_parts = [
                _stable_ids(row.customer_id for row in part.collect())
                for part in parts
            ]
        finally:
            spark.stop()
    elif backend == "sklearn_approx":
        rng = np.random.default_rng(DATA_FUSION_SEED)
        shuffled = ordered["customer_id"].astype(str).to_numpy()
        rng.shuffle(shuffled)
        test_parts = [_stable_ids(part) for part in np.array_split(shuffled, 5)]
    else:
        raise ValueError(f"Unknown Data Fusion split backend: {backend}")
    all_ids = set(ordered["customer_id"].astype(str))
    return [
        (_stable_ids(all_ids - set(test)), test)
        for test in test_parts
    ]


def berka_folds(labels: pd.DataFrame) -> list[tuple[list[str], list[str]]]:
    """Match UniTTab's repeated 70/30 random-split shape."""
    ordered = labels.sort_values("customer_id").reset_index(drop=True)
    n_train = int(np.ceil(0.7 * len(ordered)))
    folds = []
    for seed in BERKA_SEEDS:
        rng = np.random.RandomState(seed)
        permutation = rng.permutation(len(ordered))
        train = ordered.iloc[permutation[:n_train]]["customer_id"]
        test = ordered.iloc[permutation[n_train:]]["customer_id"]
        folds.append((_stable_ids(train), _stable_ids(test)))
    return folds


def prepare_benchmark(
    dataset: str,
    *,
    raw_root: Path,
    output_root: Path,
    split_backend: str = "pyspark",
) -> dict[str, Any]:
    spec = BENCHMARKS[dataset]
    dataset_root = output_root / dataset / spec.protocol
    events_path = dataset_root / "events.parquet"
    if dataset == "berka":
        events, labels, raw_hashes = normalize_berka(raw_root)
        if len(labels) != spec.n_entities:
            raise ValueError(
                f"Berka protocol requires {spec.n_entities} loans, got "
                f"{len(labels)}"
            )
        dataset_root.mkdir(parents=True, exist_ok=True)
        events.to_parquet(events_path, index=False)
        folds = berka_folds(labels)
        seeds = BERKA_SEEDS
        inner_size = 100
        backend = "numpy_random_state_protocol_match"
    else:
        labels, raw_hashes = normalize_datafusion(raw_root, events_path)
        if len(labels) != spec.n_entities:
            raise ValueError(
                f"Data Fusion protocol requires {spec.n_entities} users, got "
                f"{len(labels)}"
            )
        folds = datafusion_folds(labels, backend=split_backend)
        seeds = (DATA_FUSION_SEED,) * 5
        inner_size = 400
        backend = (
            "pyspark_3.3.3"
            if split_backend == "pyspark" else split_backend
        )
    labels.to_parquet(dataset_root / "labels.parquet", index=False)

    fold_manifests = []
    for fold_index, ((outer_train, outer_test), split_seed) in enumerate(
        zip(folds, seeds)
    ):
        inner_train, inner_validation = _inner_split(
            labels,
            outer_train,
            validation_size=inner_size,
            seed=137,
        )
        fold_root = dataset_root / f"fold_{fold_index}"
        manifest = _fold_payload(
            spec=spec,
            fold=fold_index,
            outer_train=outer_train,
            outer_test=outer_test,
            inner_train=inner_train,
            inner_validation=inner_validation,
            labels=labels,
            split_seed=split_seed,
            raw_hashes=raw_hashes,
            split_backend=backend,
        )
        for role, values in manifest["ids"].items():
            _write_ids(fold_root / f"{role}_ids.json", values)
        atomic_write_json(fold_root / "fold_manifest.json", manifest)
        fold_manifests.append(manifest)
    payload = {
        "schema_version": 1,
        "benchmark": asdict(spec),
        "events": str(events_path),
        "events_sha256": file_sha256(events_path),
        "labels": str(dataset_root / "labels.parquet"),
        "raw_file_sha256": raw_hashes,
        "fold_signatures": [
            manifest["fold_signature"] for manifest in fold_manifests
        ],
        "baselines": (
            DATA_FUSION_BASELINES if dataset == "datafusion_education"
            else BERKA_BASELINES
        ),
    }
    payload["preparation_signature"] = fingerprint(payload)
    atomic_write_json(dataset_root / "benchmark_manifest.json", payload)
    return payload

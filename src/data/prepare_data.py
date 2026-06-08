"""Download, normalize, split, and summarize benchmark datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets.load import load_dataset
from sklearn.model_selection import train_test_split

from src.data.mcc import mcc_to_text

RANDOM_STATE = 42
VAL_SIZE_FROM_REMAINING = 1.0 / 9.0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    output_dir: Path
    test_ids_path: Path
    test_id_column: str
    loader: Callable[[], tuple[pd.DataFrame, pd.DataFrame]]
    normalizer: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


CURRENCY_MAP = {
    810: "Рубль",
    643: "Рубль",
    978: "Евро",
    840: "Доллар США",
    826: "Фунт стерлингов",
    756: "Швейцарский франк",
    156: "Китайский юань",
    392: "Японская иена",
    980: "Украинская гривна",
    398: "Казахстанский тенге",
}

TRX_CATEGORY_MAP = {
    "POS": "оплата картой",
    "WD_ATM_PARTNER": "снятие наличных (банкомат партнера)",
    "WD_ATM_ROS": "снятие наличных (банкомат Росбанка)",
    "WD_ATM_OTHER": "снятие наличных (другой банк)",
    "CAT": "снятие наличных через кассу",
    "DEPOSIT": "пополнение счета",
    "C2C_OUT": "перевод на карту",
    "C2C_IN": "входящий перевод с карты",
    "MBO": "мобильный банк",
    "WEB": "интернет-банк",
    "BACK": "возврат средств",
}


def read_int_ids(path: Path, column: str) -> set[int]:
    return set(pd.read_csv(path, usecols=[column])[column].astype("int64"))


def label_table(df: pd.DataFrame) -> pd.DataFrame:
    labels = df.drop_duplicates("customer_id")[["customer_id", "label"]].reset_index(drop=True)
    assert labels["customer_id"].is_unique
    return labels


def parse_gender_datetime(values: pd.Series) -> pd.Series:
    parts = values.astype(str).str.extract(r"^(?P<day>\d+)\s+(?P<time>\d{1,2}:\d{2}:\d{2})$")
    if parts.isna().any(axis=None):
        bad = values[parts.isna().any(axis=1)].head(5).tolist()
        raise ValueError(f"Unexpected gender tr_datetime format. Examples: {bad}")
    return pd.Timestamp("2000-01-01") + pd.to_timedelta(parts["day"].astype(int), unit="D") + pd.to_timedelta(parts["time"])


def parse_rosbank_datetime(value: str) -> pd.Timestamp:
    months = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
        "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    day, month, year, time = re.match(r"^(\d{2})([A-Z]{3})(\d{2}):(\d{2}:\d{2}:\d{2})$", str(value)).groups()
    return pd.to_datetime(f"20{year}-{months[month]}-{day} {time}")


def load_gender_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = load_dataset("pytorch-lifestream/transactions-gender", "transactions_data", split="train").to_pandas()
    labels = load_dataset("pytorch-lifestream/transactions-gender", "labels", split="train").to_pandas()
    return transactions, labels


def normalize_gender(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = transactions.merge(labels[["customer_id", "gender"]], on="customer_id", how="inner")
    df = df.rename(columns={"gender": "label"})
    df["tr_datetime"] = parse_gender_datetime(df["tr_datetime"])
    df["amount"] = pd.to_numeric(df["amount"])
    df["mcc_code_desc"] = df["mcc_code"].map(mcc_to_text)
    df["label"] = df["label"].astype(int)
    return df[["customer_id", "tr_datetime", "amount", "mcc_code_desc", "label", "mcc_code", "tr_type", "term_id"]].copy()


def load_age_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = load_dataset("pytorch-lifestream/age-group-prediction", "transactions_train", split="train").to_pandas()
    labels = load_dataset("pytorch-lifestream/age-group-prediction", "train_target", split="train").to_pandas()
    return transactions, labels


def normalize_age(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = transactions.merge(labels[["client_id", "bins"]], on="client_id", how="inner")
    df = df.rename(
        columns={
            "client_id": "customer_id",
            "trans_date": "tr_datetime",
            "small_group": "mcc_code_desc",
            "amount_rur": "amount",
            "bins": "label",
        }
    )
    df["tr_datetime"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["tr_datetime"].astype(int), unit="D")
    df["mcc_code_desc"] = df["mcc_code_desc"].map(lambda group: f"operation group {int(group)}")
    df["amount"] = pd.to_numeric(df["amount"])
    df["label"] = df["label"].astype(int)
    return df[["customer_id", "tr_datetime", "amount", "mcc_code_desc", "label"]].copy()


def load_rosbank_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = load_dataset("pytorch-lifestream/rosbank-churn", "train", split="train").to_pandas()
    labels = transactions[["cl_id", "target_flag"]].drop_duplicates("cl_id")
    return transactions, labels


def normalize_rosbank(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["tr_datetime"] = df["TRDATETIME"].apply(parse_rosbank_datetime)
    df["currency_name"] = df["currency"].map(lambda code: CURRENCY_MAP[int(code)])
    df["trx_cat_ru"] = df["trx_category"].map(lambda category: TRX_CATEGORY_MAP[category])
    df["mcc_desc"] = df["MCC"].map(mcc_to_text)
    df["mcc_code_desc"] = df["mcc_desc"] + " [" + df["trx_cat_ru"] + "]"
    df = df.rename(columns={"cl_id": "customer_id", "target_flag": "label"})
    df["customer_id"] = df["customer_id"].astype(int)
    df["amount"] = pd.to_numeric(df["amount"])
    df["label"] = df["label"].astype(int)
    return df[["customer_id", "tr_datetime", "amount", "currency_name", "trx_cat_ru", "mcc_desc", "mcc_code_desc", "label"]].copy()


def split_clients(labels: pd.DataFrame, official_test_ids: set[int], val_size: float) -> tuple[set[int], set[int]]:
    assert official_test_ids <= set(labels["customer_id"])
    remaining = labels[~labels["customer_id"].isin(official_test_ids)]
    train, val = train_test_split(
        remaining,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=remaining["label"],
    )
    return set(train["customer_id"].astype(int)), set(val["customer_id"].astype(int))


def check_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, official_test_ids: set[int]) -> dict[str, int]:
    ids = {
        "train": set(train["customer_id"].astype(int)),
        "val": set(val["customer_id"].astype(int)),
        "test": set(test["customer_id"].astype(int)),
    }
    overlap = {
        "train_val": len(ids["train"] & ids["val"]),
        "train_test": len(ids["train"] & ids["test"]),
        "val_test": len(ids["val"] & ids["test"]),
    }
    assert overlap == {"train_val": 0, "train_test": 0, "val_test": 0}, overlap
    assert ids["test"] == official_test_ids, (len(official_test_ids - ids["test"]), len(ids["test"] - official_test_ids))
    return overlap


def summarize_split(df: pd.DataFrame) -> dict:
    labels = df.drop_duplicates("customer_id")["label"]
    txn_per_client = df.groupby("customer_id").size()
    amount = pd.to_numeric(df["amount"])
    return {
        "rows": int(len(df)),
        "clients": int(df["customer_id"].nunique()),
        "min_date": str(pd.to_datetime(df["tr_datetime"]).min()),
        "max_date": str(pd.to_datetime(df["tr_datetime"]).max()),
        "label_counts_by_client": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "majority_accuracy": float(labels.value_counts(normalize=True).max()),
        "transactions_per_client_mean": float(txn_per_client.mean()),
        "transactions_per_client_median": float(txn_per_client.median()),
        "amount_sum": float(amount.sum()),
        "amount_mean": float(amount.mean()),
        "amount_median": float(amount.median()),
        "amount_min": float(amount.min()),
        "amount_max": float(amount.max()),
        "top_categories_by_rows": {str(k): int(v) for k, v in df["mcc_code_desc"].value_counts().head(20).items()},
    }


def write_summary(report: dict, path: Path) -> None:
    lines = [f"Dataset: {report['dataset']}", f"Source: {report['source']}", ""]
    for split, stats in report["splits"].items():
        lines.extend(
            [
                f"[{split}]",
                f"rows: {stats['rows']}",
                f"clients: {stats['clients']}",
                f"date range: {stats['min_date']} - {stats['max_date']}",
                f"label counts: {stats['label_counts_by_client']}",
                f"majority accuracy: {stats['majority_accuracy']:.6f}",
                f"transactions/client mean: {stats['transactions_per_client_mean']:.3f}",
                f"transactions/client median: {stats['transactions_per_client_median']:.3f}",
                f"amount mean: {stats['amount_mean']:.6f}",
                f"amount median: {stats['amount_median']:.6f}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(spec: DatasetSpec, val_size: float = VAL_SIZE_FROM_REMAINING) -> dict:
    print(f"\nPreparing {spec.name}")
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    transactions, labels = spec.loader()
    data = spec.normalizer(transactions, labels)
    official_test_ids = read_int_ids(spec.test_ids_path, spec.test_id_column)
    train_ids, val_ids = split_clients(label_table(data), official_test_ids, val_size)

    train = data[data["customer_id"].isin(train_ids)].copy()
    val = data[data["customer_id"].isin(val_ids)].copy()
    test = data[data["customer_id"].isin(official_test_ids)].copy()
    overlap = check_splits(train, val, test, official_test_ids)

    train.to_csv(spec.output_dir / "train.csv", index=False)
    val.to_csv(spec.output_dir / "val.csv", index=False)
    test.to_csv(spec.output_dir / "test.csv", index=False)

    report = {
        "dataset": spec.name,
        "source": spec.source,
        "test_ids_path": str(spec.test_ids_path),
        "test_id_column": spec.test_id_column,
        "random_state": RANDOM_STATE,
        "val_size_from_remaining": val_size,
        "client_overlap": overlap,
        "splits": {"train": summarize_split(train), "val": summarize_split(val), "test": summarize_split(test)},
        "columns": list(train.columns),
    }
    with open(spec.output_dir / "split_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    write_summary(report, spec.output_dir / "split_summary.txt")

    for split in ["train", "val", "test"]:
        stats = report["splits"][split]
        print(f"  {split:<5}: rows={stats['rows']:,}, clients={stats['clients']:,}, labels={stats['label_counts_by_client']}, majority={stats['majority_accuracy']:.4f}")
    return report


def build_specs(test_ids_dir: Path = Path("data/test_ids"), data_dir: Path = Path("data")) -> dict[str, DatasetSpec]:
    return {
        "gender": DatasetSpec("gender", "pytorch-lifestream/transactions-gender", data_dir / "gender", test_ids_dir / "gender_test_ids.csv", "customer_id", load_gender_raw, normalize_gender),
        "age": DatasetSpec("age", "pytorch-lifestream/age-group-prediction", data_dir / "age", test_ids_dir / "age_test_ids.csv", "client_id", load_age_raw, normalize_age),
        "rosbank": DatasetSpec("rosbank", "pytorch-lifestream/rosbank-churn", data_dir / "rosbank", test_ids_dir / "rosbank_test_ids.csv", "cl_id", load_rosbank_raw, normalize_rosbank),
    }

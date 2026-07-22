"""
src/data/loader.py

Loads any supported dataset into the internal schema:
    customer_id, tr_datetime, amount, mcc_code_desc, label

All downstream code (aggregator, prompt_builder, lora_trainer) works
exclusively with these column names — dataset differences are isolated here.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np


def load_dataset(config: dict, split: str = "train") -> pd.DataFrame:
    """
    Load a split CSV and rename columns to the internal schema.

    Args:
        config: full pipeline config dict (from yaml)
        split:  "train", "val", or "test"

    Returns:
        DataFrame with columns: customer_id, tr_datetime, amount,
                                mcc_code_desc, label
    """
    path = config["dataset"]["splits"][split]
    df = pd.read_csv(path)

    client_filter = config["dataset"].get("client_ids_by_split", {}).get(split)
    if client_filter is not None:
        if isinstance(client_filter, (str, os.PathLike)):
            with open(Path(client_filter), encoding="utf-8") as file:
                client_filter = json.load(file)
        allowed = set(client_filter)
        source_customer_id = config["dataset"]["columns"]["customer_id"]
        if source_customer_id not in df.columns:
            raise ValueError(f"Cannot apply client filter: missing {source_customer_id}")
        df = df[df[source_customer_id].isin(allowed)].copy()
        if df.empty and allowed:
            raise ValueError(f"Client filter for split={split} selected no rows")

    col_map = config["dataset"]["columns"]
    rename = {
        col_map["customer_id"]: "customer_id",
        col_map["datetime"]:    "tr_datetime",
        col_map["amount"]:      "amount",
        col_map["category"]:    "mcc_code_desc",
        col_map["label"]:       "label",
    }
    # Only rename columns that actually need renaming
    rename = {src: dst for src, dst in rename.items() if src in df.columns and src != dst}
    df = df.rename(columns=rename)

    # Validate required columns are present
    required = {"customer_id", "tr_datetime", "amount", "mcc_code_desc", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"After renaming, columns {missing} are missing.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Check column mapping in config."
        )

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features used by the aggregator.
    Input DataFrame must already have the internal schema.

    Features added:
        is_positive / is_negative, amount_positive / amount_negative,
        hour, weekday, weekday_name, day (days since 2000-01-01),
        is_weekend, period_of_day, days_since_last_txn
    """
    df = df.copy()

    # Amount features
    df["is_positive"] = (df["amount"] > 0).astype(int)
    df["is_negative"] = (df["amount"] < 0).astype(int)
    df["amount_positive"] = df["amount"].where(df["amount"] > 0, 0)
    df["amount_negative"] = df["amount"].where(df["amount"] < 0, 0)

    # Datetime features
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    df = df.sort_values(["customer_id", "tr_datetime"], ascending=[True, True])

    df["hour"]    = df["tr_datetime"].dt.hour.fillna(0).astype(int)
    df["weekday"] = df["tr_datetime"].dt.weekday.fillna(0).astype(int)
    df["day"] = (
        df["tr_datetime"] - pd.Timestamp("2000-01-01")
    ).dt.days.fillna(0).astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["days_since_last_txn"] = df.groupby("customer_id")["day"].transform(
        lambda x: x.max() - x
    )

    weekday_map = {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}
    df["weekday_name"] = df["weekday"].map(weekday_map)

    def _hour_to_period(h):
        if 6 <= h < 12:  return "утро"
        elif 12 <= h < 18: return "день"
        elif 18 <= h < 24: return "вечер"
        else:              return "ночь"

    df["period_of_day"] = df["hour"].apply(_hour_to_period)

    return df


def load_all_splits(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience: load and add features for all three splits."""
    train = add_features(load_dataset(config, "train"))
    val   = add_features(load_dataset(config, "val"))
    test  = add_features(load_dataset(config, "test"))
    return train, val, test

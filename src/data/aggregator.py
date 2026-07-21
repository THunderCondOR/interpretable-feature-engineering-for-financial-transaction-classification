"""
src/data/aggregator.py

Builds compact textual transaction profiles used by:
- LLM explanation prompts,
- LoRA input texts,
- dataset-level summary prompts.

The Rosbank churn dataset gets a separate temporal aggregation because churn is
primarily a dynamic task: changes in activity, category diversity, operation type
shares, and recency are more important than static totals alone.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from src.data.profiles import (
    format_client_profile,
    format_robust_summary,
    robust_statistics_payload,
)


def _filter_top_percentile(d: dict, q: float = 0.8) -> dict:
    if not d:
        return {}
    threshold = np.percentile(list(d.values()), q * 100)
    return {k: v for k, v in d.items() if v >= threshold}


def _top_n(d: dict, n: int = 15) -> dict:
    return dict(sorted(d.items(), key=lambda x: -x[1])[:n])


def _round_dict(d: dict) -> dict:
    if not d:
        return {}
    max_val = np.max(np.abs(list(d.values())))
    d_sorted = dict(sorted(d.items(), key=lambda x: -x[1]))
    return {k: round(float(v), 2) if max_val < 1.0 else round(float(v)) for k, v in d_sorted.items()}


def _fmt(d: dict) -> str:
    if not d:
        return "нет выраженных категорий"
    return "\n".join(f"{k} = {v}" for k, v in d.items())


def _safe(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "нет данных"
    try:
        return str(round(float(x), 3))
    except Exception:
        return str(x)


def _share(part: float, whole: float) -> float:
    return round(float(part) / float(whole), 4) if whole else 0.0


def _label_name(config: dict, label: int) -> str:
    names = config["dataset"].get("label_names", {})
    return names.get(str(label), names.get(label, str(label)))


# ---------------------------------------------------------------------------
# Generic profile for gender / age datasets where amount can be signed.
# ---------------------------------------------------------------------------

def build_user_summary_str(client_df: pd.DataFrame, category_label: str = "категории трат") -> str:
    n_txn = len(client_df)
    n_days = max(client_df["tr_datetime"].dt.date.nunique(), 1) if client_df["tr_datetime"].notna().any() else 1
    txn_per_day = round(n_txn / n_days, 3)

    pos = client_df.loc[client_df["amount"] > 0, "amount"]
    neg = client_df.loc[client_df["amount"] < 0, "amount"]

    cat_freq = _round_dict(_filter_top_percentile(client_df["mcc_code_desc"].value_counts().to_dict(), q=0.8))
    expense_df = client_df[client_df["amount"] < 0]
    cat_amount = expense_df.groupby("mcc_code_desc")["amount"].sum().to_dict() if len(expense_df) else {}
    cat_amount = _round_dict(_filter_top_percentile(cat_amount, q=0.8)) if cat_amount else {}

    return (
        f"* Период активности: {n_days} дней\n"
        f"* Всего операций: {n_txn}\n"
        f"* Среднее число операций в день: {txn_per_day}\n"
        f"* Распределение по {category_label}:\n{_fmt(cat_freq)}\n\n"
        f"* Объем расходов по {category_label}:\n{_fmt(cat_amount)}\n\n"
        f"* Общий доход: {_safe(float(pos.sum()) if len(pos) else 0)}\n"
        f"* Средний доход на операцию: {_safe(float(pos.mean()) if len(pos) else None)}\n"
        f"* Общие расходы: {_safe(float(neg.sum()) if len(neg) else 0)}\n"
        f"* Средний расход на операцию: {_safe(float(neg.mean()) if len(neg) else None)}"
    )


# ---------------------------------------------------------------------------
# Rosbank-specific temporal churn profile.
# ---------------------------------------------------------------------------

def _period_stats(c: pd.DataFrame, name: str) -> dict:
    if c.empty:
        return {
            f"{name}_txn": 0,
            f"{name}_amount": 0.0,
            f"{name}_unique_mcc": 0,
            f"{name}_pos_share": 0.0,
            f"{name}_atm_share": 0.0,
            f"{name}_deposit_share": 0.0,
        }
    n = len(c)
    trx = c["trx_cat_ru"].value_counts() if "trx_cat_ru" in c.columns else pd.Series(dtype=int)
    atm = sum(v for k, v in trx.items() if "снятие" in str(k).lower())
    return {
        f"{name}_txn": int(n),
        f"{name}_amount": float(c["amount"].sum()),
        f"{name}_unique_mcc": int(c["mcc_code_desc"].nunique()),
        f"{name}_pos_share": _share(trx.get("оплата картой", 0), n),
        f"{name}_atm_share": _share(atm, n),
        f"{name}_deposit_share": _share(trx.get("пополнение счета", 0), n),
    }


def build_user_summary_str_rosbank(client_df: pd.DataFrame, category_label: str = "категории операций") -> str:
    c = client_df.sort_values("tr_datetime").copy()
    n_txn = len(c)
    n_days = max(c["tr_datetime"].dt.date.nunique(), 1) if c["tr_datetime"].notna().any() else 1
    n_months = max(c["tr_datetime"].dt.to_period("M").nunique(), 1) if c["tr_datetime"].notna().any() else 1

    total_amount = float(c["amount"].sum())
    avg_amount = float(c["amount"].mean()) if n_txn else 0.0
    median_amount = float(c["amount"].median()) if n_txn else 0.0
    max_amount = float(c["amount"].max()) if n_txn else 0.0

    cat_freq = _round_dict(_top_n(c["mcc_code_desc"].value_counts().to_dict(), 15))
    cat_amount = _round_dict(_top_n(c.groupby("mcc_code_desc")["amount"].sum().to_dict(), 15))

    trx_type_block = ""
    if "trx_cat_ru" in c.columns:
        trx_dist = _round_dict(c["trx_cat_ru"].value_counts().to_dict())
        trx_type_block = f"* Распределение по типам операций:\n{_fmt(trx_dist)}\n\n"

    currency_block = ""
    if "currency_name" in c.columns:
        curr_dist = c["currency_name"].value_counts().to_dict()
        curr_str = ", ".join(f"{k}: {v}" for k, v in curr_dist.items())
        currency_block = f"* Валюты операций: {curr_str}\n"

    temporal_block = ""
    if c["tr_datetime"].notna().any() and n_txn > 1:
        t_min, t_max = c["tr_datetime"].min(), c["tr_datetime"].max()
        duration = (t_max - t_min).total_seconds()
        if duration > 0:
            mid = t_min + pd.Timedelta(seconds=duration / 2)
            q1 = t_min + pd.Timedelta(seconds=duration * 0.25)
            q3 = t_max - pd.Timedelta(seconds=duration * 0.25)
            first_half = c[c["tr_datetime"] <= mid]
            second_half = c[c["tr_datetime"] > mid]
            early_quarter = c[c["tr_datetime"] <= q1]
            recent_quarter = c[c["tr_datetime"] >= q3]

            first = _period_stats(first_half, "first_half")
            second = _period_stats(second_half, "second_half")
            early = _period_stats(early_quarter, "early_quarter")
            recent = _period_stats(recent_quarter, "recent_quarter")

            txn_change = _share(second["second_half_txn"] - first["first_half_txn"], max(first["first_half_txn"], 1))
            amount_change = _share(second["second_half_amount"] - first["first_half_amount"], max(first["first_half_amount"], 1))
            diversity_change = second["second_half_unique_mcc"] - first["first_half_unique_mcc"]
            recency_ratio = round(recent["recent_quarter_txn"] / max(early["early_quarter_txn"], 1), 3)
            inactive_tail_days = int((t_max - c["tr_datetime"].max()).days) if pd.notna(t_max) else 0

            monthly_counts = c.groupby(c["tr_datetime"].dt.to_period("M")).size()
            monthly_str = ", ".join(f"{str(k)}: {int(v)}" for k, v in monthly_counts.tail(6).items())

            temporal_block = (
                f"* Динамика активности:\n"
                f"  - Операций в первой половине периода: {first['first_half_txn']}\n"
                f"  - Операций во второй половине периода: {second['second_half_txn']}\n"
                f"  - Относительное изменение числа операций: {txn_change}\n"
                f"  - Относительное изменение суммы операций: {amount_change}\n"
                f"  - Изменение разнообразия категорий: {diversity_change}\n"
                f"  - Отношение активности в последней четверти к первой: {recency_ratio}\n"
                f"  - Последние месячные количества операций: {monthly_str}\n"
            )
    
    return (
        f"* Период активности: {n_days} дней\n"
        f"* Активных месяцев: {n_months}\n"
        f"* Всего операций: {n_txn}\n"
        f"* Среднее число операций в день: {round(n_txn / n_days, 3)}\n"
        f"* Среднее число операций в месяц: {round(n_txn / n_months, 3)}\n"
        f"{currency_block}"
        f"{trx_type_block}"
        f"* Распределение по {category_label}:\n{_fmt(cat_freq)}\n\n"
        f"* Сумма операций по {category_label}:\n{_fmt(cat_amount)}\n\n"
        f"{temporal_block}"
        f"* Общая сумма операций: {_safe(total_amount)}\n"
        f"* Средняя сумма операции: {_safe(avg_amount)}\n"
        f"* Медианная сумма операции: {_safe(median_amount)}\n"
        f"* Максимальная сумма операции: {_safe(max_amount)}"
    )


def get_summary_fn(config: dict) -> Callable:
    # Retain the old two-argument callable API while making v2 profiles depend
    # on explicit dataset amount semantics.
    return lambda client_df, _category_label=None: format_client_profile(client_df, config)


# ---------------------------------------------------------------------------
# Dataset-level summary for prompts.
# ---------------------------------------------------------------------------

def _client_level_rosbank_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for cid, c in df.groupby("customer_id", sort=False):
        c = c.sort_values("tr_datetime")
        label = int(c["label"].iloc[0]) if "label" in c.columns else -1
        n = len(c)
        n_days = max(c["tr_datetime"].dt.date.nunique(), 1) if c["tr_datetime"].notna().any() else 1
        trx = c["trx_cat_ru"].value_counts() if "trx_cat_ru" in c.columns else pd.Series(dtype=int)
        atm = sum(v for k, v in trx.items() if "снятие" in str(k).lower())
        rec = {
            "customer_id": cid,
            "label": label,
            "n_txn": n,
            "active_days": n_days,
            "txn_per_day": n / n_days,
            "total_amount": float(c["amount"].sum()),
            "avg_amount": float(c["amount"].mean()) if n else 0.0,
            "unique_categories": c["mcc_code_desc"].nunique(),
            "share_pos": _share(trx.get("оплата картой", 0), n),
            "share_atm": _share(atm, n),
            "share_deposit": _share(trx.get("пополнение счета", 0), n),
            "share_c2c_out": _share(trx.get("перевод на карту", 0), n),
        }
        if c["tr_datetime"].notna().any() and n > 1:
            t_min, t_max = c["tr_datetime"].min(), c["tr_datetime"].max()
            duration = (t_max - t_min).total_seconds()
            if duration > 0:
                q1 = t_min + pd.Timedelta(seconds=duration * 0.25)
                q3 = t_max - pd.Timedelta(seconds=duration * 0.25)
                early = (c["tr_datetime"] <= q1).sum()
                recent = (c["tr_datetime"] >= q3).sum()
                rec["recency_ratio"] = recent / max(early, 1)
            else:
                rec["recency_ratio"] = 1.0
        else:
            rec["recency_ratio"] = 1.0
        records.append(rec)
    return pd.DataFrame(records)


def build_dataset_summary_str(df: pd.DataFrame, config: dict) -> str:
    labelled = df[df["label"] >= 0] if "label" in df else df
    return format_robust_summary(robust_statistics_payload(labelled, config))


def build_all_client_stats(df: pd.DataFrame, config: dict) -> list[dict]:
    label_names = config["dataset"]["label_names"]
    category_label = config["dataset"].get("category_label", "категории трат")
    summary_fn = get_summary_fn(config)
    records = []
    for cid, client_df in df.groupby("customer_id", sort=False):
        label = int(client_df["label"].iloc[0]) if "label" in client_df.columns else -1
        label_name = label_names.get(str(label), "unknown") if label >= 0 else "unknown"
        records.append({
            "customer_id": int(cid),
            "label": label,
            "label_name": label_name,
            "client_stats": summary_fn(client_df, category_label),
        })
    return records

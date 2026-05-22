"""
src/data/aggregator.py

Turns a client's transaction DataFrame into a compact text summary
(user_summary_str) and builds dataset-level statistics used in few-shot
prompts (build_dataset_summary_str).

Expects the internal schema from loader.py:
    customer_id, tr_datetime, amount, mcc_code_desc, label,
    is_positive, is_negative, period_of_day, weekday_name,
    days_since_last_txn  (added by add_features)
"""

import math
import numpy as np
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Low-level helpers (kept from original stats.py)
# ---------------------------------------------------------------------------

def filter_top_percentile(d: dict, q: float = 0.8) -> dict:
    """Keep only entries whose value is at or above the q-th percentile."""
    if not d:
        return {}
    threshold = np.percentile(list(d.values()), q * 100)
    return {k: v for k, v in d.items() if v >= threshold}


def filter_top_n(d: dict, n: int = 20) -> dict:
    """Keep top-n entries by value."""
    return dict(sorted(d.items(), key=lambda x: -x[1])[:n])


def round_dict_values(d: dict) -> dict:
    """Sort by value descending, round to int or 2dp depending on magnitude."""
    if not d:
        return {}
    max_val = np.max(np.abs(list(d.values())))
    keys = list(d.keys())
    if isinstance(keys[0], str):
        d_sorted = dict(sorted(d.items(), key=lambda x: -x[1]))
    else:
        d_sorted = dict(sorted(d.items(), key=lambda x: x[0]))
    return {
        k: round(v, 2) if max_val < 1.0 else round(v)
        for k, v in d_sorted.items()
    }


def compose_stats_str(d: dict) -> str:
    """Format a dict as 'key = value' lines."""
    return "\n".join(f"{k} = {round(v, 4)}" for k, v in d.items())


# ---------------------------------------------------------------------------
# Per-client summary
# ---------------------------------------------------------------------------

def build_user_summary_str(client_df: pd.DataFrame, category_label: str = "категории трат") -> str:
    """
    Build a compact text profile of a single client.

    Args:
        client_df:      slice of the main DataFrame for one customer_id
        category_label: dataset-specific label for mcc_code_desc column,
                        e.g. "категории MCC" for gender, "типы операций" for age.
                        Comes from config["dataset"]["category_label"].

    Returns:
        Multi-line string ready for insertion into LLM prompt.
    """
    df = client_df.copy()

    # Active period
    n_txn = len(df)
    n_days = max(df["tr_datetime"].dt.date.nunique(), 1) if df["tr_datetime"].notna().any() else 1
    txn_per_day = round(n_txn / n_days, 3)

    # Amount stats
    pos = df.loc[df["amount"] > 0, "amount"]
    neg = df.loc[df["amount"] < 0, "amount"]

    def _safe(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "Нет информации"
        return round(x)

    total_income  = _safe(float(pos.sum()) if len(pos) else 0)
    total_expense = _safe(float(neg.sum()) if len(neg) else 0)
    avg_income    = _safe(float(pos.mean()) if len(pos) else float("nan"))
    avg_expense   = _safe(float(neg.mean()) if len(neg) else float("nan"))

    # Category frequency (top 80th percentile of spending categories)
    cat_freq = df["mcc_code_desc"].value_counts().to_dict()
    cat_freq = filter_top_percentile(cat_freq, q=0.8)
    cat_freq = round_dict_values(cat_freq)

    # Category spending amounts (expenses only)
    expense_df = df[df["amount"] < 0]
    cat_amount = expense_df.groupby("mcc_code_desc")["amount"].sum().to_dict()
    cat_amount = filter_top_percentile(cat_amount, q=0.8) if cat_amount else {}
    cat_amount = round_dict_values(cat_amount)

    return (
        f"* Период активности: {n_days} дней\n"
        f"* Всего операций: {n_txn}\n"
        f"* Среднее число операций в день: {txn_per_day}\n"
        f"* Распределение по {category_label}:\n"
        f"{compose_stats_str(cat_freq)}\n\n"
        f"* Объём расходов по {category_label}:\n"
        f"{compose_stats_str(cat_amount)}\n\n"
        f"* Общий доход: {total_income}\n"
        f"* Средний доход на операцию: {avg_income}\n"
        f"* Общие расходы: {total_expense}\n"
        f"* Средний расход на операцию: {avg_expense}"
    )


# ---------------------------------------------------------------------------
# Dataset-level summary (for SUMMARY_TRANSACTIONAL_STATS in prompts)
# ---------------------------------------------------------------------------

def build_dataset_summary_str(df: pd.DataFrame, config: dict) -> str:
    """
    Build a dataset-level summary split by label groups.
    Used as SUMMARY_TRANSACTIONAL_STATS in few-shot prompts.

    Shows average transaction frequency per category per label group,
    filtered to the 90th percentile (mirrors original pipeline logic).
    """
    label_names: dict = config["dataset"]["label_names"]
    category_label: str = config["dataset"].get("category_label", "категории трат")
    lines = []

    for label_id_str, label_name in label_names.items():
        label_id = int(label_id_str)
        group_df = df[df["label"] == label_id]
        if group_df.empty:
            continue

        # Average number of transactions per category per client
        per_client = (
            group_df.groupby(["customer_id", "mcc_code_desc"])
            .size()
            .reset_index(name="count")
        )
        avg_freq = per_client.groupby("mcc_code_desc")["count"].mean().to_dict()
        avg_freq = filter_top_percentile(avg_freq, q=0.9)
        avg_freq = round_dict_values(avg_freq)

        lines.append(f"# {label_name.capitalize()}")
        lines.append(
            f"## Среднее число транзакций по {category_label} "
            f"({label_name}, 90-й перцентиль)"
        )
        lines.append(f"| {category_label.capitalize()} | Среднее число транзакций |")
        lines.append("|---|---|")
        for cat, freq in list(avg_freq.items())[:25]:
            lines.append(f"| {cat} | {freq} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build all client stats and save (used in run_pipeline.py stats step)
# ---------------------------------------------------------------------------

def build_all_client_stats(df: pd.DataFrame, config: dict) -> list[dict]:
    """
    Build user_summary_str for every client.

    Returns:
        List of dicts: {customer_id, label, label_name, client_stats}
    """
    label_names = config["dataset"]["label_names"]
    category_label = config["dataset"].get("category_label", "категории трат")
    records = []

    for cid in df["customer_id"].unique():
        client_df = df[df["customer_id"] == cid]
        label = int(client_df["label"].iloc[0])
        records.append({
            "customer_id": int(cid),
            "label": label,
            "label_name": label_names[str(label)],
            "client_stats": build_user_summary_str(client_df, category_label),
        })

    return records

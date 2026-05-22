"""
src/data/aggregator.py

Строит текстовый профиль клиента (user_summary_str) из его транзакций.
Этот профиль используется как входные данные для CoT-промптов и LoRA.

Ключевые функции:
    build_user_summary_str()         — универсальная (gender / age)
    build_user_summary_str_rosbank() — rosbank (amount всегда > 0)
    get_summary_fn(config)           — диспетчер по имени датасета

    build_dataset_summary_str()      — статистика по датасету (в промпт)
    build_all_client_stats()         — профили всех клиентов
"""

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_top_percentile(d: dict, q: float = 0.8) -> dict:
    """Оставить только записи выше q-го перцентиля по значению."""
    if not d:
        return {}
    threshold = np.percentile(list(d.values()), q * 100)
    return {k: v for k, v in d.items() if v >= threshold}


def _round_dict(d: dict) -> dict:
    """Отсортировать по убыванию, округлить значения."""
    if not d:
        return {}
    max_val = np.max(np.abs(list(d.values())))
    d_sorted = dict(sorted(d.items(), key=lambda x: -x[1]))
    return {k: round(v, 2) if max_val < 1.0 else round(v)
            for k, v in d_sorted.items()}


def _fmt(d: dict) -> str:
    return "\n".join(f"{k} = {v}" for k, v in d.items())


def _safe(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "нет данных"
    return str(round(x))


# ---------------------------------------------------------------------------
# Client summary builders
# ---------------------------------------------------------------------------

def build_user_summary_str(
    client_df: pd.DataFrame,
    category_label: str = "категории трат",
) -> str:
    """
    Универсальный текстовый профиль клиента для gender / age датасетов.

    Содержит:
    - активность (дней, транзакций, частота)
    - частотное распределение по категориям (топ 80-й перцентиль)
    - объём расходов по категориям (только отрицательные суммы)
    - общие доходы и расходы
    """
    n_txn  = len(client_df)
    n_days = max(client_df["tr_datetime"].dt.date.nunique(), 1) \
        if client_df["tr_datetime"].notna().any() else 1
    txn_per_day = round(n_txn / n_days, 3)

    pos = client_df.loc[client_df["amount"] > 0, "amount"]
    neg = client_df.loc[client_df["amount"] < 0, "amount"]

    total_income  = _safe(float(pos.sum())  if len(pos) else 0)
    total_expense = _safe(float(neg.sum())  if len(neg) else 0)
    avg_income    = _safe(float(pos.mean()) if len(pos) else float("nan"))
    avg_expense   = _safe(float(neg.mean()) if len(neg) else float("nan"))

    cat_freq = client_df["mcc_code_desc"].value_counts().to_dict()
    cat_freq = _filter_top_percentile(cat_freq, q=0.8)
    cat_freq = _round_dict(cat_freq)

    expense_df = client_df[client_df["amount"] < 0]
    cat_amount = expense_df.groupby("mcc_code_desc")["amount"].sum().to_dict() \
        if len(expense_df) else {}
    cat_amount = _filter_top_percentile(cat_amount, q=0.8) if cat_amount else {}
    cat_amount = _round_dict(cat_amount)

    return (
        f"* Период активности: {n_days} дней\n"
        f"* Всего операций: {n_txn}\n"
        f"* Среднее число операций в день: {txn_per_day}\n"
        f"* Распределение по {category_label}:\n{_fmt(cat_freq)}\n\n"
        f"* Объём расходов по {category_label}:\n{_fmt(cat_amount)}\n\n"
        f"* Общий доход: {total_income}\n"
        f"* Средний доход на операцию: {avg_income}\n"
        f"* Общие расходы: {total_expense}\n"
        f"* Средний расход на операцию: {avg_expense}"
    )


def build_user_summary_str_rosbank(
    client_df: pd.DataFrame,
    category_label: str = "категории операций",
) -> str:
    """
    Текстовый профиль клиента Росбанка для LLM.

    Отличия от gender/age:
    - amount всегда > 0 — нет разделения на доход/расход
    - есть trx_cat_ru (тип операции): POS, снятие, пополнение, переводы
    - есть currency_name: важен для валютных операций
    """
    n_txn  = len(client_df)
    n_days = max(client_df["tr_datetime"].dt.date.nunique(), 1) \
        if client_df["tr_datetime"].notna().any() else 1
    txn_per_day = round(n_txn / n_days, 3)

    total_amount = _safe(float(client_df["amount"].sum()))
    avg_amount   = _safe(float(client_df["amount"].mean()))
    max_amount   = _safe(float(client_df["amount"].max()))

    cat_freq = client_df["mcc_code_desc"].value_counts().to_dict()
    cat_freq = _filter_top_percentile(cat_freq, q=0.8)
    cat_freq = _round_dict(cat_freq)

    cat_amount = client_df.groupby("mcc_code_desc")["amount"].sum().to_dict()
    cat_amount = _filter_top_percentile(cat_amount, q=0.8) if cat_amount else {}
    cat_amount = _round_dict(cat_amount)

    trx_type_block = ""
    if "trx_cat_ru" in client_df.columns:
        trx_dist = client_df["trx_cat_ru"].value_counts().to_dict()
        trx_dist = _round_dict(trx_dist)
        trx_type_block = f"* Распределение по типам операций:\n{_fmt(trx_dist)}\n\n"

    currency_block = ""
    if "currency_name" in client_df.columns:
        curr_dist = client_df["currency_name"].value_counts().to_dict()
        if len(curr_dist) > 1 or list(curr_dist.keys()) != ["Рубль"]:
            curr_str = ", ".join(f"{k}: {v}" for k, v in curr_dist.items())
            currency_block = f"* Валюты операций: {curr_str}\n"

    return (
        f"* Период активности: {n_days} дней\n"
        f"* Всего операций: {n_txn}\n"
        f"* Среднее число операций в день: {txn_per_day}\n"
        f"{currency_block}"
        f"{trx_type_block}"
        f"* Распределение по {category_label}:\n{_fmt(cat_freq)}\n\n"
        f"* Объём трат по {category_label}:\n{_fmt(cat_amount)}\n\n"
        f"* Общая сумма операций: {total_amount}\n"
        f"* Средняя сумма операции: {avg_amount}\n"
        f"* Максимальная сумма операции: {max_amount}"
    )


def get_summary_fn(config: dict):
    """Вернуть нужную функцию построения профиля клиента по имени датасета."""
    if config["dataset"]["name"] == "rosbank":
        return build_user_summary_str_rosbank
    return build_user_summary_str


# ---------------------------------------------------------------------------
# Dataset-level summary (for SUMMARY_TRANSACTIONAL_STATS in prompts)
# ---------------------------------------------------------------------------

def build_dataset_summary_str(df: pd.DataFrame, config: dict) -> str:
    """
    Строит датасет-уровневую статистику по каждому классу.
    Вставляется как SUMMARY_TRANSACTIONAL_STATS в промпт.
    """
    label_names:    dict = config["dataset"]["label_names"]
    category_label: str  = config["dataset"].get("category_label", "категории трат")
    lines = []

    for label_id_str, label_name in label_names.items():
        label_id = int(label_id_str)
        group_df = df[df["label"] == label_id]
        if group_df.empty:
            continue

        per_client = (
            group_df.groupby(["customer_id", "mcc_code_desc"])
            .size()
            .reset_index(name="count")
        )
        avg_freq = per_client.groupby("mcc_code_desc")["count"].mean().to_dict()
        avg_freq = _filter_top_percentile(avg_freq, q=0.9)
        avg_freq = _round_dict(avg_freq)

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
# Per-client stats (list of dicts written to clients_stats.jsonl)
# ---------------------------------------------------------------------------

def build_all_client_stats(df: pd.DataFrame, config: dict) -> list:
    """
    Строит user_summary_str для каждого клиента.
    Возвращает список dict: {customer_id, label, label_name, client_stats}
    """
    label_names    = config["dataset"]["label_names"]
    category_label = config["dataset"].get("category_label", "категории трат")
    summary_fn     = get_summary_fn(config)
    records = []

    for cid in df["customer_id"].unique():
        client_df = df[df["customer_id"] == cid]
        label = int(client_df["label"].iloc[0])
        records.append({
            "customer_id": int(cid),
            "label":       label,
            "label_name":  label_names.get(str(label), label_names.get(label, str(label))),
            "client_stats": summary_fn(client_df, category_label),
        })

    return records
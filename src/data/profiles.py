"""Dataset-aware client profiles and robust train-only descriptive statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUANTILES = ((0.05, "p05"), (0.25, "q1"), (0.50, "median"), (0.75, "q3"), (0.95, "p95"))


def amount_semantics(config: dict) -> str:
    """Return explicit amount semantics, with safe backwards-compatible defaults."""
    configured = config["dataset"].get("amount_semantics")
    if configured:
        return configured
    return {
        "gender": "signed_cashflow",
        "age": "unsigned_transaction_value",
        "rosbank": "typed_transaction_value",
    }.get(config["dataset"]["name"], "unsigned_transaction_value")


def _share(part: float, whole: float) -> float:
    return float(part) / float(whole) if whole else 0.0


def _observation_end(config: dict, train_df: pd.DataFrame | None = None) -> pd.Timestamp | None:
    value = config["dataset"].get("observation_end")
    if value:
        return pd.Timestamp(value)
    if train_df is not None and train_df["tr_datetime"].notna().any():
        return pd.Timestamp(train_df["tr_datetime"].max())
    return None


def client_numeric_profile(
    client_df: pd.DataFrame,
    config: dict,
    *,
    observation_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Build a numeric profile without assigning unsupported meaning to amounts."""
    c = client_df.sort_values("tr_datetime")
    valid_dates = c["tr_datetime"].dropna()
    n_txn = len(c)
    active_days = int(valid_dates.dt.normalize().nunique()) if len(valid_dates) else 0
    calendar_span = int((valid_dates.max().normalize() - valid_dates.min().normalize()).days + 1) if len(valid_dates) else 0
    amounts = pd.to_numeric(c["amount"], errors="coerce").dropna()
    profile: dict[str, Any] = {
        "customer_id": c["customer_id"].iloc[0],
        "label": int(c["label"].iloc[0]) if "label" in c and pd.notna(c["label"].iloc[0]) else -1,
        "transactions_per_client": int(n_txn),
        "active_days": active_days,
        "calendar_span_days": calendar_span,
        "transactions_per_active_day": _share(n_txn, active_days),
        "unique_categories": int(c["mcc_code_desc"].nunique()),
    }

    semantics = amount_semantics(config)
    if semantics == "signed_cashflow":
        inflow = amounts[amounts > 0]
        outflow = -amounts[amounts < 0]
        profile.update({
            "total_inflow": float(inflow.sum()),
            "median_inflow": float(inflow.median()) if len(inflow) else 0.0,
            "total_outflow": float(outflow.sum()),
            "median_outflow": float(outflow.median()) if len(outflow) else 0.0,
            "inflow_operation_share": _share(len(inflow), n_txn),
            "outflow_operation_share": _share(len(outflow), n_txn),
        })
    else:
        values = amounts.abs()
        profile.update({
            "total_transaction_value": float(values.sum()),
            "median_transaction_value": float(values.median()) if len(values) else 0.0,
            "p95_transaction_value": float(values.quantile(0.95)) if len(values) else 0.0,
        })

    if semantics == "typed_transaction_value" and "trx_cat_ru" in c:
        types = c["trx_cat_ru"].fillna("").astype(str).str.lower()
        profile.update({
            "card_payment_share": float(types.str.contains("оплата картой", regex=False).mean()),
            "cash_withdrawal_share": float(types.str.contains("снятие", regex=False).mean()),
            "deposit_share": float(types.str.contains("пополнение", regex=False).mean()),
            "outgoing_transfer_share": float(types.str.contains("перевод на карту", regex=False).mean()),
        })
        end = observation_end or _observation_end(config)
        profile["recency_days"] = float((end - valid_dates.max()).total_seconds() / 86400) if end is not None and len(valid_dates) else np.nan
        if len(valid_dates) > 1:
            midpoint = valid_dates.min() + (valid_dates.max() - valid_dates.min()) / 2
            first = int((c["tr_datetime"] <= midpoint).sum())
            second = int((c["tr_datetime"] > midpoint).sum())
            profile["second_to_first_activity_ratio"] = _share(second, first)
        else:
            profile["second_to_first_activity_ratio"] = 0.0
    return profile


def client_feature_frame(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aggregate once per client; the caller must pass the intended train split."""
    end = _observation_end(config, df)
    rows = [client_numeric_profile(group, config, observation_end=end) for _, group in df.groupby("customer_id", sort=False)]
    return pd.DataFrame(rows)


def _top_categories(client_df: pd.DataFrame, *, n: int = 12) -> list[tuple[str, int, float]]:
    counts = client_df["mcc_code_desc"].value_counts().head(n)
    total = max(len(client_df), 1)
    return [(str(category), int(count), float(count / total)) for category, count in counts.items()]


def format_client_profile(client_df: pd.DataFrame, config: dict) -> str:
    """Format evidence-only client facts for prompts."""
    p = client_numeric_profile(client_df, config)
    category_label = config["dataset"].get("category_label", "категории операций")
    lines = [
        f"* Всего операций: {p['transactions_per_client']}",
        f"* Активных дней: {p['active_days']}",
        f"* Календарный охват: {p['calendar_span_days']} дней",
        f"* Операций на активный день: {p['transactions_per_active_day']:.3f}",
        f"* Уникальных категорий: {p['unique_categories']}",
        f"* Наблюдаемые top-{min(12, p['unique_categories'])} {category_label}:",
    ]
    lines.extend(f"  - {cat}: {count} операций ({share:.1%})" for cat, count, share in _top_categories(client_df))
    lines.append("* Важно: отсутствие категории в top-k не означает отсутствие таких операций у клиента.")

    semantics = amount_semantics(config)
    if semantics == "signed_cashflow":
        expenses = client_df.loc[client_df["amount"] < 0].copy()
        if not expenses.empty:
            expenses["outflow"] = -expenses["amount"]
            by_category = expenses.groupby("mcc_code_desc")["outflow"].sum().sort_values(ascending=False).head(12)
            lines.append("* Категории расходов по абсолютной величине оттока:")
            lines.extend(f"  - {cat}: {float(value):.2f}" for cat, value in by_category.items())
        lines.extend([
            f"* Общий приток: {p['total_inflow']:.2f}",
            f"* Медианный приток на операцию: {p['median_inflow']:.2f}",
            f"* Общий отток (положительная величина): {p['total_outflow']:.2f}",
            f"* Медианный отток на операцию: {p['median_outflow']:.2f}",
        ])
    else:
        lines.extend([
            f"* Общая величина операций: {p['total_transaction_value']:.2f}",
            f"* Медианная величина операции: {p['median_transaction_value']:.2f}",
            f"* 95-й перцентиль величины операции: {p['p95_transaction_value']:.2f}",
        ])
    if semantics == "typed_transaction_value":
        lines.extend([
            f"* Доля оплат картой: {p['card_payment_share']:.1%}",
            f"* Доля снятий наличных: {p['cash_withdrawal_share']:.1%}",
            f"* Доля пополнений: {p['deposit_share']:.1%}",
            f"* Доля исходящих переводов: {p['outgoing_transfer_share']:.1%}",
            f"* Дней от последней операции до конца окна наблюдения: {p['recency_days']:.1f}",
            f"* Отношение активности второй половины окна к первой: {p['second_to_first_activity_ratio']:.3f}",
        ])
    return "\n".join(lines)


def robust_statistics_payload(df: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Describe untrimmed client-level train observations using robust summaries."""
    clients = client_feature_frame(df, config)
    label_names = config["dataset"].get("label_names", {})
    numeric = [column for column in clients.select_dtypes(include=[np.number]).columns if column not in {"label", "customer_id"}]
    payload: dict[str, Any] = {
        "scope": "training_split_only",
        "amount_semantics": amount_semantics(config),
        "outlier_handling": "untrimmed_observations",
        "n_clients": int(clients.shape[0]),
        "classes": {},
    }
    prevalence = (df.groupby(["customer_id", "mcc_code_desc"]).size().reset_index(name="count"))
    top_categories = (prevalence.groupby("mcc_code_desc")["customer_id"].nunique().sort_values(ascending=False).head(20).index.tolist())
    labels_by_client = df.groupby("customer_id", sort=False)["label"].first()
    for label, group in clients.groupby("label", sort=True):
        class_payload: dict[str, Any] = {"label": int(label), "name": label_names.get(str(int(label)), str(int(label))), "n_clients": int(len(group)), "metrics": {}, "categories": []}
        for column in numeric:
            values = group[column].dropna()
            if values.empty:
                continue
            class_payload["metrics"][column] = {name: float(values.quantile(q)) for q, name in QUANTILES}
        class_ids = set(group["customer_id"].tolist())
        class_counts = prevalence[prevalence["customer_id"].isin(class_ids)]
        totals = class_counts.groupby("customer_id")["count"].sum()
        for category in top_categories:
            rows = class_counts[class_counts["mcc_code_desc"] == category].set_index("customer_id")["count"]
            shares = (rows / totals.loc[rows.index]).dropna()
            class_payload["categories"].append({
                "category": str(category),
                "client_prevalence": _share(rows.index.nunique(), len(class_ids)),
                "median_share_among_users": float(shares.median()) if len(shares) else 0.0,
            })
        payload["classes"][str(int(label))] = class_payload
    payload["label_hash_input"] = {str(k): int(v) for k, v in labels_by_client.value_counts().sort_index().items()}
    return payload


def format_robust_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Train-only robust dataset summary",
        f"Clients: {payload['n_clients']}",
        f"Amount semantics: {payload['amount_semantics']}",
        "Outliers: observations are not removed; P5/Q1/median/Q3/P95 are reported.",
    ]
    for class_payload in payload["classes"].values():
        lines.extend(["", f"## {class_payload['name']}", f"Clients: {class_payload['n_clients']}"])
        for metric, values in class_payload["metrics"].items():
            lines.append(f"* {metric}: P5={values['p05']:.3f}, Q1={values['q1']:.3f}, median={values['median']:.3f}, Q3={values['q3']:.3f}, P95={values['p95']:.3f}")
        lines.append("* Category prevalence and median within-client share among users:")
        lines.extend(f"  - {row['category']}: prevalence={row['client_prevalence']:.1%}, median share={row['median_share_among_users']:.1%}" for row in class_payload["categories"])
    return "\n".join(lines)


def export_robust_statistics(payload: dict[str, Any], output_dir: str | Path, stem: str = "robust_statistics") -> dict[str, Path]:
    """Export one payload as JSON, long CSV, Markdown, and LaTeX."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for class_payload in payload["classes"].values():
        for metric, values in class_payload["metrics"].items():
            rows.extend({"label": class_payload["label"], "class_name": class_payload["name"], "kind": "metric", "item": metric, "statistic": stat, "value": value} for stat, value in values.items())
        for category in class_payload["categories"]:
            rows.extend([
                {"label": class_payload["label"], "class_name": class_payload["name"], "kind": "category", "item": category["category"], "statistic": "client_prevalence", "value": category["client_prevalence"]},
                {"label": class_payload["label"], "class_name": class_payload["name"], "kind": "category", "item": category["category"], "statistic": "median_share_among_users", "value": category["median_share_among_users"]},
            ])
    frame = pd.DataFrame(rows)
    paths = {suffix: output / f"{stem}.{suffix}" for suffix in ("json", "csv", "md", "tex")}
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.to_csv(paths["csv"], index=False)
    markdown = format_robust_summary(payload)
    paths["md"].write_text(markdown + "\n", encoding="utf-8")
    latex_rows = ["\\begin{tabular}{lllrr}", "Class & Type & Item & Statistic & Value \\\\ ", "\\hline"]
    for row in rows:
        safe = lambda value: str(value).replace("_", "\\_").replace("&", "\\&")
        latex_rows.append(f"{safe(row['class_name'])} & {safe(row['kind'])} & {safe(row['item'])} & {safe(row['statistic'])} & {float(row['value']):.4g} \\\\ ")
    latex_rows.append("\\end{tabular}")
    paths["tex"].write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    return paths

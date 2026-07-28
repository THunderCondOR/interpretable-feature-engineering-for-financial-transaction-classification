"""Dataset-aware client profiles and robust train-only descriptive statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.prompt_locale import (
    amount_semantics_display,
    english_category,
    metric_display_name,
)


QUANTILES = (
    (0.05, "p05"),
    (0.25, "q1"),
    (0.50, "median"),
    (0.75, "q3"),
    (0.95, "p95"),
)


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


def _currency_bucket(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"RUR", "USD", "EUR"}:
        return normalized
    return "unknown"


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
    currency_column = next(
        (
            column for column in ("currency_name", "currency_code")
            if column in c.columns
        ),
        None,
    )
    if currency_column is not None:
        # Arrow-backed Parquet columns are loaded as categoricals to keep the
        # multi-million-row Data Fusion table compact.  Convert the small
        # per-client slice before filling so "unknown" need not be inserted
        # into the global categorical dictionary.
        currency = c[currency_column].astype("string").fillna("unknown")
        profile.update({
            "unique_currencies": int(currency.nunique()),
            "dominant_currency_share": float(
                currency.value_counts(normalize=True).iloc[0]
            ) if len(currency) else 0.0,
        })

    semantics = amount_semantics(config)
    if semantics == "signed_cashflow":
        if config["dataset"].get("currency_aware_amounts") and currency_column:
            currency = c[currency_column].map(_currency_bucket)
            numeric_amounts = pd.to_numeric(c["amount"], errors="coerce")
            for bucket in ("RUR", "USD", "EUR", "unknown"):
                bucket_amounts = numeric_amounts[currency.eq(bucket)].dropna()
                inflow = bucket_amounts[bucket_amounts > 0]
                outflow = -bucket_amounts[bucket_amounts < 0]
                prefix = bucket.lower()
                profile.update({
                    f"{prefix}_operation_count": int(len(bucket_amounts)),
                    f"{prefix}_operation_share": _share(
                        len(bucket_amounts), n_txn
                    ),
                    f"{prefix}_total_inflow": float(inflow.sum()),
                    f"{prefix}_median_inflow": (
                        float(inflow.median()) if len(inflow) else 0.0
                    ),
                    f"{prefix}_total_outflow": float(outflow.sum()),
                    f"{prefix}_median_outflow": (
                        float(outflow.median()) if len(outflow) else 0.0
                    ),
                })
        else:
            inflow = amounts[amounts > 0]
            outflow = -amounts[amounts < 0]
            profile.update({
                "total_inflow": float(inflow.sum()),
                "mean_inflow": float(inflow.mean()) if len(inflow) else 0.0,
                "median_inflow": float(inflow.median()) if len(inflow) else 0.0,
                "total_outflow": float(outflow.sum()),
                "mean_outflow": float(outflow.mean()) if len(outflow) else 0.0,
                "median_outflow": float(outflow.median()) if len(outflow) else 0.0,
                "inflow_operation_share": _share(len(inflow), n_txn),
                "outflow_operation_share": _share(len(outflow), n_txn),
            })
    else:
        values = amounts.abs()
        profile.update({
            "total_transaction_value": float(values.sum()),
            "mean_transaction_value": float(values.mean()) if len(values) else 0.0,
            "median_transaction_value": float(values.median()) if len(values) else 0.0,
            "p95_transaction_value": float(values.quantile(0.95)) if len(values) else 0.0,
        })

    if semantics == "typed_transaction_value" and "trx_cat_ru" in c:
        types = c["trx_cat_ru"].astype("string").fillna("").str.lower()
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
    elif semantics == "typed_transaction_value" and "operation_type" in c:
        types = c["operation_type"].astype("string").fillna("").str.lower()
        profile.update({
            "credit_operation_share": float(types.eq("credit").mean()),
            "debit_operation_share": float(types.eq("debit").mean()),
        })
        if "balance" in c:
            balance = pd.to_numeric(c["balance"], errors="coerce").dropna()
            profile.update({
                "mean_balance": float(balance.mean()) if len(balance) else 0.0,
                "median_balance": float(balance.median()) if len(balance) else 0.0,
                "minimum_balance": float(balance.min()) if len(balance) else 0.0,
                "negative_balance_share": float((balance < 0).mean())
                if len(balance)
                else 0.0,
            })
    return profile


def client_feature_frame(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Aggregate once per client; the caller must pass the intended train split."""
    end = _observation_end(config, df)
    rows = [
        client_numeric_profile(group, config, observation_end=end)
        for _, group in df.groupby(
            "customer_id", sort=False, observed=True
        )
    ]
    return pd.DataFrame(rows)


def _top_categories(client_df: pd.DataFrame, *, n: int = 12) -> list[tuple[str, int, float]]:
    counts = client_df["mcc_code_desc"].value_counts()
    counts = counts[counts > 0].head(n)
    total = max(len(client_df), 1)
    return [
        (english_category(category), int(count), float(count / total))
        for category, count in counts.items()
    ]


def format_client_profile(client_df: pd.DataFrame, config: dict) -> str:
    """Format evidence-only client facts for prompts."""
    p = client_numeric_profile(client_df, config)
    category_label = config["dataset"].get("category_label", "transaction categories")
    lines = [
        f"* Total transactions: {p['transactions_per_client']}",
        f"* Active days: {p['active_days']}",
        f"* Calendar span: {p['calendar_span_days']} days",
        f"* Transactions per active day: {p['transactions_per_active_day']:.3f}",
        f"* Unique transaction categories: {p['unique_categories']}",
        f"* Observed top-{min(12, p['unique_categories'])} {category_label}:",
    ]
    lines.extend(
        f"  - {cat}: {count} transactions ({share:.1%})"
        for cat, count, share in _top_categories(client_df)
    )
    lines.append(
        "* Coverage note: this is a top-k list; an omitted category is not "
        "evidence that the client never used it."
    )
    currency_column = next(
        (
            column for column in ("currency_name", "currency_code")
            if column in client_df.columns
        ),
        None,
    )
    if currency_column is not None:
        counts = (
            client_df[currency_column]
            .astype("string")
            .fillna("unknown")
            .value_counts()
        )
        lines.append(
            "* Transaction currencies: "
            + ", ".join(
                f"{name}: {int(count)} ({count / len(client_df):.1%})"
                for name, count in counts.items()
            )
        )

    semantics = amount_semantics(config)
    if semantics == "signed_cashflow":
        expenses = client_df.loc[client_df["amount"] < 0].copy()
        if not expenses.empty:
            expenses["outflow"] = -expenses["amount"]
            if (
                config["dataset"].get("currency_aware_amounts")
                and currency_column
            ):
                expenses["_currency_bucket"] = expenses[currency_column].map(
                    _currency_bucket
                )
                lines.append(
                    "* Leading outflow categories within each currency:"
                )
                for bucket, group in expenses.groupby(
                    "_currency_bucket", observed=True
                ):
                    by_category = (
                        group.groupby("mcc_code_desc", observed=True)["outflow"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(6)
                    )
                    lines.extend(
                        f"  - [{bucket}] {english_category(cat)}: "
                        f"{float(value):.2f}"
                        for cat, value in by_category.items()
                    )
            else:
                by_category = (
                    expenses.groupby("mcc_code_desc", observed=True)["outflow"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(12)
                )
                lines.append(
                    "* Outflow categories by total positive outflow magnitude:"
                )
                lines.extend(
                    f"  - {english_category(cat)}: {float(value):.2f}"
                    for cat, value in by_category.items()
                )
        if config["dataset"].get("currency_aware_amounts") and currency_column:
            lines.append(
                "* Currency-specific cash flow (amounts are never summed "
                "across currencies):"
            )
            for bucket in ("RUR", "USD", "EUR", "unknown"):
                prefix = bucket.lower()
                count = int(p[f"{prefix}_operation_count"])
                if not count:
                    continue
                lines.append(
                    f"  - {bucket}: {count} operations "
                    f"({p[f'{prefix}_operation_share']:.1%}); "
                    f"total inflow={p[f'{prefix}_total_inflow']:.2f}; "
                    f"median inflow={p[f'{prefix}_median_inflow']:.2f}; "
                    f"total outflow={p[f'{prefix}_total_outflow']:.2f}; "
                    f"median outflow={p[f'{prefix}_median_outflow']:.2f}"
                )
        else:
            lines.extend([
                f"* Total inflow: {p['total_inflow']:.2f}",
                f"* Mean inflow per inflow operation: {p['mean_inflow']:.2f}",
                f"* Median inflow per inflow operation: {p['median_inflow']:.2f}",
                f"* Total outflow (positive magnitude): {p['total_outflow']:.2f}",
                f"* Mean outflow per outflow operation: {p['mean_outflow']:.2f}",
                f"* Median outflow per outflow operation: {p['median_outflow']:.2f}",
            ])
    else:
        lines.extend([
            f"* Total transaction value: {p['total_transaction_value']:.2f}",
            f"* Mean transaction value: {p['mean_transaction_value']:.2f}",
            f"* Median transaction value: {p['median_transaction_value']:.2f}",
            f"* P95 transaction value: {p['p95_transaction_value']:.2f}",
        ])
    if semantics == "typed_transaction_value":
        if "credit_operation_share" in p:
            lines.extend([
                f"* Share of credit operations: {p['credit_operation_share']:.1%}",
                f"* Share of debit operations: {p['debit_operation_share']:.1%}",
                f"* Mean observed account balance: {p['mean_balance']:.2f}",
                f"* Median observed account balance: {p['median_balance']:.2f}",
                f"* Minimum observed account balance: {p['minimum_balance']:.2f}",
                f"* Share of observations with negative balance: {p['negative_balance_share']:.1%}",
            ])
        else:
            lines.extend([
                f"* Share of card payments: {p['card_payment_share']:.1%}",
                f"* Share of cash withdrawals: {p['cash_withdrawal_share']:.1%}",
                f"* Share of account deposits: {p['deposit_share']:.1%}",
                f"* Share of outgoing card-to-card transfers: {p['outgoing_transfer_share']:.1%}",
                f"* Days from the last transaction to the observation end: {p['recency_days']:.1f}",
                f"* Second-half / first-half activity ratio: {p['second_to_first_activity_ratio']:.3f}",
            ])
    return "\n".join(lines)


def robust_statistics_payload(df: pd.DataFrame, config: dict) -> dict[str, Any]:
    """Describe untrimmed client-level train observations with mean and tails."""
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
    prevalence = (
        df.groupby(
            ["customer_id", "mcc_code_desc"], observed=True
        ).size().reset_index(name="count")
    )
    top_categories = (
        prevalence.groupby("mcc_code_desc", observed=True)["customer_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )
    labels_by_client = df.groupby(
        "customer_id", sort=False, observed=True
    )["label"].first()
    for label, group in clients.groupby("label", sort=True):
        class_payload: dict[str, Any] = {"label": int(label), "name": label_names.get(str(int(label)), str(int(label))), "n_clients": int(len(group)), "metrics": {}, "categories": []}
        for column in numeric:
            values = group[column].dropna()
            if values.empty:
                continue
            class_payload["metrics"][column] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                **{
                    name: float(values.quantile(q))
                    for q, name in QUANTILES
                },
            }
        class_ids = set(group["customer_id"].tolist())
        class_counts = prevalence[prevalence["customer_id"].isin(class_ids)]
        totals = class_counts.groupby("customer_id", observed=True)["count"].sum()
        for category in top_categories:
            rows = class_counts[class_counts["mcc_code_desc"] == category].set_index("customer_id")["count"]
            shares = (rows / totals.loc[rows.index]).dropna()
            class_payload["categories"].append({
                "category": english_category(category),
                "category_original": str(category),
                "client_prevalence": _share(rows.index.nunique(), len(class_ids)),
                # Missing category rows are real zeros, not missing values.
                "mean_transaction_count_all_clients": float(
                    rows.reindex(list(class_ids), fill_value=0).mean()
                ),
                "median_share_among_users": float(shares.median()) if len(shares) else 0.0,
            })
        payload["classes"][str(int(label))] = class_payload
    payload["label_hash_input"] = {str(k): int(v) for k, v in labels_by_client.value_counts().sort_index().items()}
    return payload


def format_robust_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Training-split class reference",
        f"Total training clients: {payload['n_clients']}",
        f"Amount semantics: {amount_semantics_display(payload['amount_semantics'])}",
        (
            "Observations are untrimmed. When the mean and median differ "
            "substantially, use the median, IQR, and P5-P95 range to account "
            "for heavy tails."
        ),
    ]
    for class_payload in payload["classes"].values():
        lines.extend(["", f"## {class_payload['name']}", f"Clients: {class_payload['n_clients']}"])
        for metric, values in class_payload["metrics"].items():
            lines.append(
                f"* {metric_display_name(metric)}: mean={values['mean']:.3f}, "
                f"median={values['median']:.3f}, "
                f"IQR=[{values['q1']:.3f}, {values['q3']:.3f}], "
                f"P5-P95=[{values['p05']:.3f}, {values['p95']:.3f}]"
            )
        lines.append(
            "* Most prevalent transaction categories. Each row reports client "
            "prevalence; mean transactions per class client including zeros; "
            "and median within-client share among clients who used the category:"
        )
        lines.extend(
            f"  - {row['category']}: used by {row['client_prevalence']:.1%}; "
            f"mean count including zeros={row['mean_transaction_count_all_clients']:.3f}; "
            f"median share among users={row['median_share_among_users']:.1%}"
            for row in class_payload["categories"]
        )
    return "\n".join(lines)


def format_legacy_mean_category_summary(df: pd.DataFrame, config: dict) -> str:
    """Corrected legacy-format category means used only by the neutral pilot."""
    lines = ["# Training-split category-frequency summary (legacy format)"]
    category_label = config["dataset"].get("category_label", "transaction categories")
    for label, name in config["dataset"].get("label_names", {}).items():
        group = df[df["label"] == int(label)]
        counts = (
            group.groupby(["customer_id", "mcc_code_desc"], observed=True)
            .size()
            .reset_index(name="count")
        )
        means = (
            counts.groupby("mcc_code_desc", observed=True)["count"]
            .mean()
            .sort_values(ascending=False)
            .head(25)
        )
        lines.extend(["", f"## {name}", f"| {category_label} | mean count among clients using category |", "|---|---:|"])
        lines.extend(
            f"| {english_category(category)} | {value:.3f} |"
            for category, value in means.items()
        )
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
                {"label": class_payload["label"], "class_name": class_payload["name"], "kind": "category", "item": category["category"], "statistic": "mean_transaction_count_all_clients", "value": category["mean_transaction_count_all_clients"]},
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

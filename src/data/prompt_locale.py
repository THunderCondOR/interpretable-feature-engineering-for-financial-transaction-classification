"""Deterministic English rendering for model-facing transaction text.

Prepared split files intentionally retain their historical values.  This module
is the only translation boundary used by prompts, few-shot demonstrations, and
LLM-facing summaries.
"""

from __future__ import annotations

import re
from typing import Iterable

from src.data.mcc import MCC_TO_DESC, MCC_TO_DESC_EN


PROMPT_LANGUAGE = "en"
CATEGORY_MAPPING_VERSION = "en_v1"
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

_MCC_RU_TO_EN = {
    russian: MCC_TO_DESC_EN[code]
    for code, russian in MCC_TO_DESC.items()
    if code in MCC_TO_DESC_EN
}

_OPERATION_RU_TO_EN = {
    "оплата картой": "card payment",
    "снятие наличных (банкомат партнера)": "cash withdrawal — partner ATM",
    "снятие наличных (банкомат Росбанка)": "cash withdrawal — Rosbank ATM",
    "снятие наличных (другой банк)": "cash withdrawal — other bank ATM",
    "снятие наличных через кассу": "cash withdrawal — branch counter",
    "пополнение счета": "account deposit",
    "перевод на карту": "outgoing card-to-card transfer",
    "входящий перевод с карты": "incoming card-to-card transfer",
    "мобильный банк": "mobile banking",
    "интернет-банк": "online banking",
    "возврат средств": "refund",
    "возврат транзакции": "transaction reversal",
}

_CURRENCY_RU_TO_EN = {
    "Рубль": "Russian ruble",
    "Евро": "Euro",
    "Доллар США": "US dollar",
    "Фунт стерлингов": "British pound",
    "Швейцарский франк": "Swiss franc",
    "Китайский юань": "Chinese yuan",
    "Японская иена": "Japanese yen",
    "Украинская гривна": "Ukrainian hryvnia",
    "Казахстанский тенге": "Kazakhstani tenge",
    "Марокканский дирхам": "Moroccan dirham",
}

METRIC_DISPLAY_NAMES = {
    "transactions_per_client": "Transactions per client",
    "active_days": "Active days",
    "calendar_span_days": "Calendar span in days",
    "transactions_per_active_day": "Transactions per active day",
    "unique_categories": "Unique transaction categories",
    "total_inflow": "Total inflow",
    "mean_inflow": "Mean inflow per inflow operation",
    "median_inflow": "Median inflow per inflow operation",
    "total_outflow": "Total outflow",
    "mean_outflow": "Mean outflow per outflow operation",
    "median_outflow": "Median outflow per outflow operation",
    "inflow_operation_share": "Share of inflow operations",
    "outflow_operation_share": "Share of outflow operations",
    "total_transaction_value": "Total transaction value",
    "mean_transaction_value": "Mean transaction value",
    "median_transaction_value": "Median transaction value",
    "p95_transaction_value": "Per-client P95 transaction value",
    "card_payment_share": "Share of card payments",
    "cash_withdrawal_share": "Share of cash withdrawals",
    "deposit_share": "Share of account deposits",
    "outgoing_transfer_share": "Share of outgoing card-to-card transfers",
    "recency_days": "Days from the last transaction to the observation end",
    "second_to_first_activity_ratio": "Second-half / first-half activity ratio",
    "credit_operation_share": "Share of credit operations",
    "debit_operation_share": "Share of debit operations",
    "mean_balance": "Mean observed account balance",
    "median_balance": "Median observed account balance",
    "minimum_balance": "Minimum observed account balance",
    "negative_balance_share": "Share of observations with negative balance",
    "unique_currencies": "Unique transaction currencies",
    "dominant_currency_share": "Share of operations in the dominant currency",
}

AMOUNT_SEMANTICS_DISPLAY = {
    "signed_cashflow": (
        "Signed cash flow: positive amounts are inflows; negative amounts are "
        "outflows and are displayed by positive magnitude."
    ),
    "unsigned_transaction_value": (
        "Unsigned transaction value: amount is transaction value, not income "
        "or expense direction."
    ),
    "typed_transaction_value": (
        "Transaction value with direction defined by operation type, not by "
        "the amount sign."
    ),
    "signed_direction": (
        "Signed transaction direction in the native currency. Positive and "
        "negative values are described separately without assigning "
        "unsupported accounting semantics, and currencies are never summed."
    ),
    "typed_unsigned_transaction_value": (
        "Positive transaction value whose behavioral meaning is defined by "
        "the recorded transaction type."
    ),
}


def contains_cyrillic(text: object) -> bool:
    return bool(CYRILLIC_RE.search(str(text)))


def assert_english_model_text(text: object, *, context: str) -> None:
    value = str(text)
    match = CYRILLIC_RE.search(value)
    if match:
        start = max(0, match.start() - 35)
        end = min(len(value), match.start() + 70)
        snippet = value[start:end].replace("\n", " ")
        raise ValueError(
            f"Cyrillic leaked into English model-facing text ({context}): {snippet!r}"
        )


def assert_all_english(values: Iterable[object], *, context: str) -> None:
    for index, value in enumerate(values):
        assert_english_model_text(value, context=f"{context}[{index}]")


def english_operation_type(value: object) -> str:
    text = str(value).strip()
    if text in _OPERATION_RU_TO_EN:
        return _OPERATION_RU_TO_EN[text]
    if not contains_cyrillic(text):
        return text
    raise ValueError(f"Unknown Russian operation type for English prompt: {text!r}")


def english_currency(value: object) -> str:
    text = str(value).strip()
    if text in _CURRENCY_RU_TO_EN:
        return _CURRENCY_RU_TO_EN[text]
    unknown_code = re.fullmatch(r"Валюта\s+(\d+)", text)
    if unknown_code:
        return f"Currency code {unknown_code.group(1)}"
    if not contains_cyrillic(text):
        return text
    raise ValueError(f"Unknown Russian currency for English prompt: {text!r}")


def english_category(value: object) -> str:
    """Translate a prepared MCC display value, including Rosbank composites."""
    text = str(value).strip()
    composite = re.fullmatch(r"(.+?)\s+\[(.+)]", text)
    if composite:
        category = english_category(composite.group(1))
        operation = english_operation_type(composite.group(2))
        return f"{category} [{operation}]"
    if text in _MCC_RU_TO_EN:
        return _MCC_RU_TO_EN[text]
    if re.fullmatch(r"operation group \d+", text, flags=re.IGNORECASE):
        return text.lower()
    if re.fullmatch(r"MCC \d+", text, flags=re.IGNORECASE):
        return text.upper()
    if not contains_cyrillic(text):
        return text
    raise ValueError(f"Unknown Russian category for English prompt: {text!r}")


def metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, str(metric).replace("_", " ").capitalize())


def amount_semantics_display(semantics: str) -> str:
    try:
        return AMOUNT_SEMANTICS_DISPLAY[semantics]
    except KeyError as exc:
        raise ValueError(f"Unknown amount semantics: {semantics!r}") from exc

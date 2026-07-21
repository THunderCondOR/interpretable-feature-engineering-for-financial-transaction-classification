from pathlib import Path

import pandas as pd

from src.data.profiles import client_numeric_profile, export_robust_statistics, format_client_profile, robust_statistics_payload


def _config(name: str, semantics: str, **dataset):
    return {"dataset": {"name": name, "amount_semantics": semantics, "category_label": "категории операций", "label_names": {"0": "zero", "1": "one"}, **dataset}}


def _frame(rows):
    frame = pd.DataFrame(rows)
    frame["tr_datetime"] = pd.to_datetime(frame["tr_datetime"])
    return frame


def test_gender_outflow_is_positive_and_sorted_by_absolute_magnitude():
    frame = _frame([
        {"customer_id": 1, "label": 0, "tr_datetime": "2024-01-01", "amount": -10, "mcc_code_desc": "small"},
        {"customer_id": 1, "label": 0, "tr_datetime": "2024-01-03", "amount": -100, "mcc_code_desc": "large"},
        {"customer_id": 1, "label": 0, "tr_datetime": "2024-01-03", "amount": 25, "mcc_code_desc": "inflow"},
    ])
    config = _config("gender", "signed_cashflow")
    profile = client_numeric_profile(frame, config)
    text = format_client_profile(frame, config)
    assert profile["total_outflow"] == 110
    assert profile["active_days"] == 2
    assert profile["calendar_span_days"] == 3
    assert text.index("large: 100.00") < text.index("small: 10.00")
    assert "Общий отток (положительная величина): 110.00" in text


def test_age_uses_neutral_transaction_value_language():
    frame = _frame([
        {"customer_id": 3, "label": 1, "tr_datetime": "2024-01-01", "amount": 12, "mcc_code_desc": "A"},
        {"customer_id": 3, "label": 1, "tr_datetime": "2024-01-02", "amount": 30, "mcc_code_desc": "B"},
    ])
    text = format_client_profile(frame, _config("age", "unsigned_transaction_value"))
    assert "Общая величина операций" in text
    assert "доход" not in text.lower()
    assert "расход" not in text.lower()


def test_rosbank_recency_uses_fixed_observation_end_and_operation_types():
    frame = _frame([
        {"customer_id": 7, "label": 1, "tr_datetime": "2024-01-01", "amount": 10, "mcc_code_desc": "A", "trx_cat_ru": "оплата картой"},
        {"customer_id": 7, "label": 1, "tr_datetime": "2024-01-08", "amount": 20, "mcc_code_desc": "B", "trx_cat_ru": "снятие наличных"},
    ])
    profile = client_numeric_profile(frame, _config("rosbank", "typed_transaction_value", observation_end="2024-01-11"))
    assert profile["recency_days"] == 3
    assert profile["card_payment_share"] == 0.5
    assert profile["cash_withdrawal_share"] == 0.5


def test_robust_stats_are_client_level_untrimmed_and_export_all_formats(tmp_path: Path):
    frame = _frame([
        {"customer_id": cid, "label": cid % 2, "tr_datetime": "2024-01-01", "amount": amount, "mcc_code_desc": "A"}
        for cid, amount in enumerate([1, 2, 3, 4, 1000], start=1)
    ])
    payload = robust_statistics_payload(frame, _config("age", "unsigned_transaction_value"))
    paths = export_robust_statistics(payload, tmp_path)
    assert payload["scope"] == "training_split_only"
    assert payload["outlier_handling"] == "untrimmed_observations"
    assert payload["n_clients"] == 5
    assert set(paths) == {"json", "csv", "md", "tex"}
    assert all(path.exists() for path in paths.values())
    assert "P5/Q1/median/Q3/P95" in paths["md"].read_text(encoding="utf-8")

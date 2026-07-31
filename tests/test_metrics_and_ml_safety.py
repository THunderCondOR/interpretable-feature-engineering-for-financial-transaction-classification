import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_gender_v2 import _metric
from src.models.ml_baseline import (
    build_handcrafted_features,
    build_llm_profile_features,
    evaluate,
    merge_feature_frames,
    prediction_artifact_complete,
    validate_client_feature_frame,
)
from src.pipeline.llm_eval import summarize_prediction_rows


def test_llm_metrics_exclude_rows_with_error_even_when_prediction_exists():
    result = summarize_prediction_rows(
        [
            {"customer_id": 1, "label": 0, "predicted": 0},
            {"customer_id": 2, "label": 1, "predicted": 0, "error": "truncated"},
        ],
        split="test",
    )
    assert result["n_scored"] == 1
    assert result["n_errors"] == 1
    assert result["n_skipped"] == 1
    assert result["coverage"] == 0.5
    assert result["accuracy"] == 1.0


def test_ml_evaluator_omits_binary_only_metrics_for_multiclass_targets():
    class MulticlassModel:
        def predict(self, features):
            return np.asarray([0, 2, 1, 2])

        def predict_proba(self, features):
            return np.asarray(
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.1, 0.8],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.2, 0.7],
                ]
            )

    metrics = evaluate(
        MulticlassModel(),
        np.zeros((4, 1)),
        np.asarray([0, 1, 1, 2]),
    )
    assert "positive_f1" not in metrics
    assert "roc_auc" not in metrics
    assert metrics["accuracy"] == 0.75
    assert metrics["f1_macro"] > 0


def test_gender_metric_recomputes_legacy_on_exact_pilot_ids(tmp_path):
    metrics = tmp_path / "llm_metrics_val.json"
    metrics.write_text(json.dumps({"balanced_accuracy": 0.0}), encoding="utf-8")
    rows = [
        {"customer_id": 1, "label": 0, "predicted": 0},
        {"customer_id": 2, "label": 1, "predicted": 1},
        {"customer_id": 3, "label": 1, "predicted": 0},
    ]
    (tmp_path / "explanations_val.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result = _metric(metrics, [1, 2])
    assert result["n_rows"] == 2
    assert result["balanced_accuracy"] == 1.0


def test_cot_features_require_exact_ids_and_labels():
    canonical = pd.DataFrame({"customer_id": [1, 2], "label": [0, 1]})
    valid = pd.DataFrame(
        {"customer_id": [2, 1], "label": [1, 0], "cot_a": [1, 0]}
    )
    aligned = validate_client_feature_frame(
        valid, canonical, split="train", source="CoT features"
    )
    assert aligned["customer_id"].tolist() == [1, 2]

    with pytest.raises(ValueError, match="client IDs differ"):
        validate_client_feature_frame(
            valid.iloc[:1], canonical, split="train", source="CoT features"
        )
    wrong_label = valid.copy()
    wrong_label.loc[wrong_label["customer_id"] == 1, "label"] = 1
    with pytest.raises(ValueError, match="labels differ"):
        validate_client_feature_frame(
            wrong_label, canonical, split="train", source="CoT features"
        )
    with pytest.raises(ValueError, match="different client IDs"):
        merge_feature_frames(valid, valid.iloc[:1])


def test_ml_resume_requires_every_prediction_and_probability_column():
    frame = pd.DataFrame({"customer_id": [1, 2], "label": [0, 1]})
    frames = {"train": frame, "val": frame, "test": frame}
    records = []
    for classifier in ("xgboost", "decision_tree"):
        for split in frames:
            for customer_id, label in ((1, 0), (2, 1)):
                records.append(
                    {
                        "feature_set": "cot",
                        "classifier": classifier,
                        "seed": 17,
                        "split": split,
                        "customer_id": customer_id,
                        "label": label,
                        "prediction": label,
                        "probability_0": 1.0 if label == 0 else 0.0,
                        "probability_1": 1.0 if label == 1 else 0.0,
                    }
                )
    assert prediction_artifact_complete(
        records,
        feature_set="cot",
        seeds=[17],
        frames=frames,
        num_labels=2,
    )
    incomplete = [dict(row) for row in records]
    incomplete[0].pop("probability_1")
    assert not prediction_artifact_complete(
        incomplete,
        feature_set="cot",
        seeds=[17],
        frames=frames,
        num_labels=2,
    )


def test_age_handcrafted_uses_unsigned_transaction_value_names():
    frame = pd.DataFrame(
        {
            "customer_id": [1, 1],
            "label": [0, 0],
            "amount": [10.0, 30.0],
            "tr_datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "mcc_code_desc": ["a", "b"],
        }
    )
    features = build_handcrafted_features(
        frame,
        {
            "dataset": {
                "name": "age",
                "amount_semantics": "unsigned_transaction_value",
            }
        },
    )
    assert features.loc[0, "total_transaction_value"] == 40.0
    assert features.loc[0, "median_transaction_value"] == 20.0
    assert not any(
        "income" in column or "expense" in column
        for column in features.columns
    )


def test_llm_profile_features_match_client_prompt_facts_without_class_stats():
    frame = pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "label": [0, 0, 1],
            "amount": [-10.0, -30.0, -5.0],
            "tr_datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-03", "2024-01-02"]
            ),
            "mcc_code_desc": ["food", "travel", "food"],
        }
    )
    features = build_llm_profile_features(
        frame,
        {
            "dataset": {
                "name": "gender",
                "amount_semantics": "signed_cashflow",
            }
        },
    ).set_index("customer_id")

    assert features.loc[1, "profile__transactions_per_client"] == 2
    assert features.loc[1, "profile__calendar_span_days"] == 3
    assert features.loc[1, "profile__median_outflow"] == 20
    assert features.loc[1, "profile__mcc_food_share"] == 0.5
    assert features.loc[1, "profile__mcc_travel_outflow"] == 30
    assert "label" in features.columns
    assert not any(
        "class" in column or "reference" in column
        for column in features.columns
    )

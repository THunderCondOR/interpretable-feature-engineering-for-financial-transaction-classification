import json

import pytest

from scripts.run_rosbank_backend_ml_comparison import (
    comparison_rows,
    metric_summary,
)


def test_metric_summary_extracts_test_means_and_variation():
    metrics = {
        "cot": {
            "xgboost": {
                "summary": {
                    "test": {
                        "accuracy": {"mean": 0.7, "sd": 0.01},
                        "balanced_accuracy": {"mean": 0.65, "sd": 0.02},
                        "f1_macro": {"mean": 0.64, "sd": 0.03},
                        "roc_auc": {"mean": 0.75, "sd": 0.01},
                        "mcc": {"mean": 0.3, "sd": 0.04},
                    }
                }
            }
        }
    }
    summary = metric_summary(metrics, "cot")
    assert summary["accuracy"]["mean"] == 0.7
    assert summary["balanced_accuracy"]["sd"] == 0.02
    assert "mcc" not in summary


def test_comparison_rows_report_cluster_seed_mean_and_legacy_delta(tmp_path):
    for model in ("qwen", "gpt_oss"):
        root = tmp_path / "rosbank" / model / "seed_17"
        root.mkdir(parents=True)
        payload = {
            feature_set: {
                "xgboost": {
                    "summary": {
                        "test": {
                            metric: {"mean": 0.5}
                            for metric in (
                                "accuracy",
                                "balanced_accuracy",
                                "f1_macro",
                                "roc_auc",
                            )
                        }
                    }
                }
            }
            for feature_set in ("cot", "concat")
        }
        (root / "ml_metrics.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    results = [
        {
            "model": model,
            "metrics": {
                feature_set: {
                    metric: {"mean": value}
                    for metric in (
                        "accuracy",
                        "balanced_accuracy",
                        "f1_macro",
                        "roc_auc",
                    )
                }
                for feature_set in ("cot", "concat")
            },
        }
        for model in ("qwen", "gpt_oss")
        for value in (0.6, 0.8)
    ]
    rows = comparison_rows(tmp_path, results)
    balanced = next(
        row for row in rows
        if row["model"] == "qwen"
        and row["feature_set"] == "cot"
        and row["metric"] == "balanced_accuracy"
    )
    assert balanced["minibatch_cluster_seed_mean"] == 0.7
    assert balanced["delta"] == pytest.approx(0.2)

import json

import pytest

from scripts.summarize_cv_benchmarks import aggregate, ml_rows
from src.data.benchmark_registry import BENCHMARKS


def test_aggregate_uses_fold_means_not_seed_runs():
    rows = [
        {
            "dataset": "berka",
            "model": "qwen",
            "experiment": "cot",
            "classifier": "xgboost",
            "fold": fold,
            "seed": seed,
            "metrics": {"positive_f1": value},
        }
        for fold, values in ((0, (0.2, 0.4)), (1, (0.8, 1.0)))
        for seed, value in zip((17, 101), values)
    ]
    result = aggregate(rows)[0]
    assert result["seed_fold_runs"] == 4
    assert result["metrics"]["positive_f1"]["n"] == 2
    assert result["metrics"]["positive_f1"]["mean"] == pytest.approx(0.6)


def test_aggregate_deduplicates_model_independent_feature_sets():
    base = {
        "dataset": "berka",
        "experiment": "standard",
        "classifier": "catboost",
        "fold": 0,
        "seed": 17,
        "metrics": {"positive_f1": 0.7},
    }
    result = aggregate(
        [{**base, "model": "qwen"}, {**base, "model": "gpt_oss"}]
    )[0]
    assert result["model"] == "shared"
    assert result["seed_fold_runs"] == 1

    with pytest.raises(ValueError, match="metrics disagree"):
        aggregate(
            [
                {**base, "model": "qwen"},
                {
                    **base,
                    "model": "gpt_oss",
                    "metrics": {"positive_f1": 0.6},
                },
            ]
        )


def test_ml_rows_includes_optional_boosters(tmp_path):
    spec = BENCHMARKS["berka"]
    root = tmp_path / "berka" / spec.protocol / "fold_0" / "qwen"
    root.mkdir(parents=True)
    (root / "ml_metrics.json").write_text("{}", encoding="utf-8")
    (root / "optional_booster_metrics.json").write_text(
        json.dumps(
            {
                "feature_sets": {
                    "all_features": {
                        "catboost": {
                            "runs": {
                                "17": {"positive_f1": 0.7},
                                "101": {"positive_f1": 0.8},
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rows = ml_rows("berka", "qwen", derived_root=tmp_path)
    assert {(row["classifier"], row["seed"]) for row in rows} == {
        ("catboost", 17),
        ("catboost", 101),
    }

import json

from scripts.run_cv_fidelity import aggregate_results, fidelity_cells


def _metrics(value):
    return {
        "hard_agreement": value,
        "probability_mae": 1.0 - value,
        "probability_rmse": 1.0 - value,
        "jensen_shannon_divergence": 1.0 - value,
        "agree_and_correct": value / 2,
        "agree_and_wrong": value / 2,
        "teacher_only_correct": (1.0 - value) / 2,
        "surrogate_only_correct": (1.0 - value) / 2,
    }


def test_cv_fidelity_cells_keep_models_and_folds_separate():
    cells = fidelity_cells(folds=(0, 1), models=("qwen", "gpt_oss"))
    assert cells == [
        (0, "qwen"), (0, "gpt_oss"),
        (1, "qwen"), (1, "gpt_oss"),
    ]
    assert all("union" not in model for _, model in cells)


def test_cv_fidelity_aggregation_reports_fold_mean_and_sample_sd(tmp_path):
    for fold, value in ((0, 0.6), (1, 0.8)):
        root = tmp_path / f"fold_{fold}" / "qwen"
        root.mkdir(parents=True)
        payload = {
            "surrogate_selection": {
                "selected_surrogate": "logistic_regression",
            },
            "results": {
                family: {"test": _metrics(value)}
                for family in (
                    "logistic_regression", "xgboost", "shallow_tree"
                )
            },
        }
        (root / "fidelity_metrics.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    result = aggregate_results(tmp_path, ("qwen",), (0, 1))
    selected = next(
        row for row in result["summary"]
        if row["surrogate"] == "validation_selected"
    )
    assert selected["n_folds"] == 2
    assert selected["hard_agreement"]["mean"] == 0.7
    assert 0.14 < selected["hard_agreement"]["sd"] < 0.15
    assert result["claim_spaces"] == "separate_per_source_model"
    assert (tmp_path / "fidelity_by_fold.csv").is_file()
    assert (tmp_path / "fidelity_summary.json").is_file()

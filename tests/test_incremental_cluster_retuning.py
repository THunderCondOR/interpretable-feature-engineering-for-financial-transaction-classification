import numpy as np

from scripts.run_incremental_cluster_retuning import (
    choose_smallest_near_best,
    metric_name,
    metrics_from_probability,
    residual_feature_ranking,
    select_threshold,
)


def test_dataset_native_metric_is_not_hardcoded():
    assert metric_name({"dataset": {"name": "berka", "metric": "accuracy"}}) == "positive_f1"
    assert metric_name({"dataset": {"name": "rosbank", "metric": "accuracy"}}) == "roc_auc"
    assert metric_name({"dataset": {"name": "custom", "metric": "balanced_accuracy"}}) == "balanced_accuracy"


def test_residual_ranking_prefers_incremental_signal():
    residual = np.array([-1.0, -0.8, 0.8, 1.0])
    features = np.array([
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]
    ])
    assert residual_feature_ranking(features, residual, ["wrong", "incremental"])[0] == "incremental"


def test_k_zero_wins_inside_tie_margin():
    rows = [
        {"n_cluster_features": 0, "family": "xgboost", "validation_primary_score": 0.800},
        {"n_cluster_features": 20, "family": "xgboost", "validation_primary_score": 0.801},
        {"n_cluster_features": 50, "family": "lightgbm", "validation_primary_score": 0.790},
    ]
    assert choose_smallest_near_best(rows, primary="balanced_accuracy", margin=0.002)["n_cluster_features"] == 0


def test_positive_f1_threshold_is_selected_on_validation():
    y = np.array([0, 0, 0, 1, 1])
    probability = np.array([0.05, 0.10, 0.20, 0.25, 0.35])
    threshold, selected = select_threshold(y, probability, "positive_f1")
    default = metrics_from_probability(y, probability, threshold=0.5)
    assert threshold < 0.5
    assert selected["positive_f1"] > default["positive_f1"]


def test_multiclass_metrics_do_not_apply_binary_threshold():
    y = np.array([0, 1, 2])
    probability = np.eye(3)
    metrics = metrics_from_probability(y, probability)
    assert metrics["threshold"] is None
    assert metrics["balanced_accuracy"] == 1.0

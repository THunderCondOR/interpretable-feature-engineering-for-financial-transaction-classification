import numpy as np

from scripts.run_datafusion_ml_retuning import (
    probability_metrics,
    select_balanced_threshold,
)


def test_validation_threshold_recovers_minority_without_test_access():
    labels = np.array([0, 0, 0, 0, 1, 1])
    probabilities = np.array([0.01, 0.02, 0.10, 0.20, 0.15, 0.30])
    selected = select_balanced_threshold(labels, probabilities)
    metrics = probability_metrics(labels, probabilities, selected["threshold"])
    assert selected["threshold"] != 0.5
    assert metrics["balanced_accuracy"] > 0.5
    assert metrics["confusion_matrix"][1][1] > 0


def test_probability_metrics_use_explicit_frozen_threshold():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.3, 0.8])
    low = probability_metrics(labels, probabilities, 0.25)
    high = probability_metrics(labels, probabilities, 0.5)
    assert low["roc_auc"] == high["roc_auc"]
    assert low["confusion_matrix"] != high["confusion_matrix"]

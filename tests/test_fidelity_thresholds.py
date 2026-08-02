import numpy as np

from scripts.run_fidelity_analysis import (
    select_binary_surrogate_threshold,
    select_binary_teacher_threshold,
    thresholded_binary_fidelity,
)


def _two_class(positive):
    positive = np.asarray(positive, dtype=float)
    return np.column_stack([1.0 - positive, positive])


def test_teacher_threshold_avoids_majority_collapse_on_validation():
    labels = np.array([0, 0, 0, 1, 1])
    probabilities = _two_class([0.01, 0.05, 0.20, 0.15, 0.30])
    selected = select_binary_teacher_threshold(probabilities, labels)
    assert selected["threshold"] < 0.5
    assert selected["balanced_accuracy"] > 0.5
    assert selected["positive_rate"] > 0.0


def test_surrogate_threshold_is_frozen_to_teacher_decisions():
    teacher = _two_class([0.1, 0.2, 0.7, 0.8])
    surrogate = _two_class([0.05, 0.10, 0.30, 0.40])
    selected = select_binary_surrogate_threshold(teacher, surrogate, 0.5)
    metrics = thresholded_binary_fidelity(
        teacher, surrogate, np.array([0, 0, 1, 1]),
        teacher_threshold=0.5, surrogate_threshold=selected["threshold"],
    )
    assert selected["threshold"] < 0.5
    assert metrics["hard_agreement"] == 1.0
    assert metrics["teacher_positive_rate"] == 0.5

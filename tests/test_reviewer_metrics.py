import numpy as np
import pandas as pd

from src.evaluation.reviewer_metrics import (
    adjudicate_grounding,
    cluster_occlusion,
    clustering_agreement,
    cross_model_matching,
    grounding_summary,
    surrogate_fidelity,
)


def test_two_judge_disagreement_has_no_fake_majority():
    assert adjudicate_grounding(["supported", "unsupported"]) == ("disagreement", False)
    assert adjudicate_grounding(["supported", "supported"]) == ("supported", True)


def test_grounding_reports_kappa_and_client_bootstrap():
    rows = pd.DataFrame([
        {"sample_id": "a", "customer_id": 1, "judge_name": "j1", "verdict": "supported"},
        {"sample_id": "a", "customer_id": 1, "judge_name": "j2", "verdict": "supported"},
        {"sample_id": "b", "customer_id": 2, "judge_name": "j1", "verdict": "unsupported"},
        {"sample_id": "b", "customer_id": 2, "judge_name": "j2", "verdict": "partially_supported"},
    ])
    items, summary = grounding_summary(rows, bootstrap_samples=50)
    assert set(items["verdict"]) == {"supported", "disagreement"}
    assert summary["disagreement_share"] == 0.5
    assert summary["cohen_kappa"] is not None
    assert summary["verdicts"]["supported"]["ci_low"] is not None


def test_clustering_metrics_keep_axis_and_match_models_one_to_one():
    agreement = clustering_agreement({"seed17": np.array([0, 0, 1]), "seed101": np.array([0, 0, 1])}, "seed")
    assert agreement.iloc[0]["axis"] == "seed"
    assert agreement.iloc[0]["ari"] == 1
    match = cross_model_matching(np.eye(2), np.array([[0.99, 0.01], [0.01, 0.99]]), [10, 2], [9, 3])
    assert match["mutual_nearest_pairs"] == 2
    assert match["one_to_one_mean_cosine"] > 0.99


def test_surrogate_fidelity_and_cluster_occlusion():
    metrics = surrogate_fidelity(
        [[0.9, 0.1], [0.2, 0.8]], [[0.8, 0.2], [0.6, 0.4]], [0, 1]
    )
    assert metrics["hard_agreement"] == 0.5
    assert metrics["teacher_only_correct"] == 0.5

    def predict_proba(x):
        score = np.clip(0.2 + 0.3 * x[:, 0] + 0.2 * x[:, 1], 0, 1)
        return np.column_stack([1 - score, score])

    result = cluster_occlusion(predict_proba, np.array([1, 1, 0]), [0, 1])
    assert result["predicted_class"] == 1
    assert result["comprehensiveness"] > 0
    assert result["sufficiency_probability"] == result["base_probability"]

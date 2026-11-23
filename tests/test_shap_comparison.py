import numpy as np

from interpretable_features.analysis.shap_comparison import (
    choose_local_example,
    raw_label,
    readable_claim,
)


def test_claim_label_removes_repeated_subject_prefix():
    assert readable_claim("the client's frequent cash withdrawals") == "Frequent cash withdrawals"


def test_raw_feature_name_is_preserved():
    assert raw_label("category_6011_amount_share") == "category 6011 amount share"


def test_raw_feature_has_fair_readable_alias():
    assert raw_label("calendar_span_days") == "Observation-window length"


def test_encoded_mcc_feature_has_fair_readable_alias():
    assert (
        raw_label("mcc_Снятие_наличных_[снятие_наличных_(другой_банк)]_sum")
        == "Total cash withdrawals from\nother banks"
    )


def test_local_example_is_median_fidelity_not_maximum_confidence():
    index, audit = choose_local_example(
        ["a", "b", "c"],
        np.array([1, 1, 1]),
        np.array([0.81, 0.90, 0.99]),
        np.array([0.80, 0.85, 0.81]),
    )
    assert index == 1
    assert audit["fallback_used"] is False
    assert audit["eligible_clients"] == 3


def test_local_example_can_require_both_contribution_directions():
    contributions = np.array([[1.0, 0.0], [1.0, -0.2], [-0.3, 0.4]])
    index, audit = choose_local_example(
        ["a", "b", "c"],
        np.array([1, 1, 1]),
        np.array([0.9, 0.9, 0.9]),
        np.array([0.9, 0.9, 0.9]),
        contributions,
        contributions,
    )
    assert index in {1, 2}
    assert "positive and negative" in audit["rule"]

import pandas as pd

from interpretable_features.evaluation.cluster_cleanup import (
    low_information_claim,
    select_clean_cluster_features,
)


def _meta(feature, medoid, clients, compactness=0.02):
    return {
        "feature": feature,
        "medoid": medoid,
        "examples": [medoid],
        "unique_clients": clients,
        "assigned_unique_clients": clients,
        "compactness_mean_distance": compactness,
    }


def test_claim_filter_rejects_meta_boilerplate_but_keeps_observation():
    assert low_information_claim(
        "The client's overall spending profile aligns with a moderate profile"
    )
    assert not low_information_claim(
        "The client has frequent cash withdrawals and a low median balance"
    )


def test_cluster_cleanup_uses_train_geometry_and_removes_redundancy():
    train = pd.DataFrame(
        {
            "customer_id": range(20),
            "label": [0, 1] * 10,
            "cot_good": [1] * 10 + [0] * 10,
            "cot_duplicate": [1] * 10 + [0] * 10,
            "cot_tiny": [1] * 5 + [0] * 15,
            "cot_generic": [0, 1] * 10,
        }
    )
    metadata = [
        _meta("cot_good", "The client frequently makes cash withdrawals", 10),
        _meta("cot_duplicate", "The client often makes cash withdrawals", 10),
        _meta("cot_tiny", "The client has recurring transfer activity", 5),
        _meta("cot_generic", "The client's spending profile aligns with a moderate profile", 10),
    ]
    selected, report = select_clean_cluster_features(
        train, metadata, variant="full_clean", compactness_quantile=1.0
    )
    assert len(selected) == 1
    assert selected[0] in {"cot_good", "cot_duplicate"}
    assert report["train_only"] is True
    assert report["removed_features"] == 3


def test_current_variant_preserves_all_columns():
    train = pd.DataFrame({"customer_id": [1], "label": [0], "cot_a": [1]})
    selected, report = select_clean_cluster_features(train, [], variant="current")
    assert selected == ["cot_a"]
    assert report["removed_features"] == 0

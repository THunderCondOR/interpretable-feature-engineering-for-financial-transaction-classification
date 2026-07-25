from scripts.run_selected_concat_queue import choose_smallest_near_best


def test_selected_concat_can_reject_all_cluster_features():
    rows = [
        {"n_cluster_features": 0, "validation_balanced_accuracy": 0.80},
        {"n_cluster_features": 5, "validation_balanced_accuracy": 0.803},
        {"n_cluster_features": 10, "validation_balanced_accuracy": 0.801},
    ]
    assert choose_smallest_near_best(rows, 0.005)["n_cluster_features"] == 0


def test_selected_concat_keeps_clear_incremental_gain():
    rows = [
        {"n_cluster_features": 0, "validation_balanced_accuracy": 0.80},
        {"n_cluster_features": 5, "validation_balanced_accuracy": 0.806},
        {"n_cluster_features": 10, "validation_balanced_accuracy": 0.814},
        {"n_cluster_features": 20, "validation_balanced_accuracy": 0.816},
    ]
    assert choose_smallest_near_best(rows, 0.005)["n_cluster_features"] == 10

from scripts.run_datafusion_cluster_auc_sweep import choose


def test_auc_selection_uses_near_best_simplest_representation():
    rows = [
        {"validation_concat_roc_auc": .700, "n_features": 100, "n_clusters": 400,
         "encoding": "raw_count"},
        {"validation_concat_roc_auc": .699, "n_features": 25, "n_clusters": 100,
         "encoding": "binary"},
        {"validation_concat_roc_auc": .696, "n_features": 10, "n_clusters": 100,
         "encoding": "binary"},
    ]
    assert choose(rows, .002)["n_features"] == 25

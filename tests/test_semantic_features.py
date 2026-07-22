import numpy as np
import pandas as pd

from src.pipeline.semantic_features import claim_occurrences, fit_semantic_space, transform_semantic_space


def _record(cid, label, *claims):
    return {"customer_id": cid, "label": label, "claims": list(claims)}


def _embedder(seen):
    def embed(texts, model_name=None):
        seen.append(list(texts))
        return np.asarray([
            [1.0, 0.02] if "alpha" in text else [0.02, 1.0]
            for text in texts
        ])
    return embed


def _config(encoding="binary", selection=False):
    return {
        "clustering": {
            "embedding_model": "synthetic",
            "n_clusters": 2,
            "min_client_coverage": 2,
            "max_train_distance": 1.0,
            "max_assign_distance": 0.2,
            "feature_encoding": encoding,
        },
        "feature_selection": {"enabled": selection, "seed": 17},
    }


def test_embedding_deduplicates_text_but_preserves_occurrences():
    records = [_record(1, 0, "alpha claim", "alpha claim"), _record(2, 1, "beta claim")]
    assert len(claim_occurrences(records)) == 3
    seen = []
    config = _config()
    config["clustering"]["min_client_coverage"] = 1
    fit_semantic_space(config, records, embedder=_embedder(seen))
    assert len(seen[0]) == 2


def test_cluster_formation_does_not_use_train_labels():
    records = [
        _record(1, 0, "alpha one"), _record(2, 0, "alpha two"),
        _record(3, 1, "beta one"), _record(4, 1, "beta two"),
    ]
    flipped = [{**row, "label": 1 - row["label"]} for row in records]
    first = fit_semantic_space(_config(), records, embedder=_embedder([]))
    second = fit_semantic_space(_config(), flipped, embedder=_embedder([]))
    assert first["formation_signature"] == second["formation_signature"]
    assert first["feature_names"] == second["feature_names"]
    np.testing.assert_allclose(first["centroids"], second["centroids"])


def test_val_test_labels_cannot_change_frozen_features():
    train = [
        _record(1, 0, "alpha one"), _record(2, 1, "alpha two"),
        _record(3, 0, "beta one"), _record(4, 1, "beta two"),
    ]
    model = fit_semantic_space(_config(), train, embedder=_embedder([]))
    target = [_record(10, 0, "alpha target", "alpha target"), _record(11, 1, "beta target")]
    relabelled = [{**row, "label": 9} for row in target]
    first = transform_semantic_space(_config(), target, model, embedder=_embedder([]))
    second = transform_semantic_space(_config(), relabelled, model, embedder=_embedder([]))
    pd.testing.assert_frame_equal(first.drop(columns="label"), second.drop(columns="label"))
    feature_columns = [column for column in first if column.startswith("cot_")]
    assert first.loc[0, feature_columns].sum() == 1.0


def test_feature_encodings_control_length_signal():
    train = [
        _record(1, 0, "alpha one"), _record(2, 1, "alpha two"),
        _record(3, 0, "beta one"), _record(4, 1, "beta two"),
    ]
    target = [_record(10, 0, "alpha x", "alpha y", "alpha z")]
    binary_model = fit_semantic_space(_config("binary"), train, embedder=_embedder([]))
    raw_model = {**binary_model, "settings": {**binary_model["settings"], "feature_encoding": "raw_count"}}
    binary = transform_semantic_space(_config(), target, binary_model, embedder=_embedder([]))
    raw = transform_semantic_space(_config(), target, raw_model, embedder=_embedder([]))
    features = binary_model["feature_names"]
    assert binary[features].to_numpy().sum() == 1
    assert raw[features].to_numpy().sum() == 3

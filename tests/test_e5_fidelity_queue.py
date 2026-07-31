import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_e5_rebuild_suite import (
    BACKEND_CANDIDATES,
    build_jobs,
)
from scripts.run_cluster_quality_rebuild import (
    expose_cell,
    selected_rosbank_sweep_root,
)
from scripts.run_fidelity_analysis import (
    bootstrap_fidelity,
    main as fidelity_main,
    ranked_occlusion,
)
from scripts.run_fidelity_suite import (
    SEED_PAIRS,
    reusable_teacher_selection,
    semantic_union_features,
    semantic_union_mapping,
    union_features,
)
from src.experiments.artifacts import atomic_write_json, files_fingerprint
from src.models import ml_baseline


def _frame(feature):
    return pd.DataFrame({
        "customer_id": [1, 2],
        "label": [0, 1],
        feature: [1.0, 2.0],
    })


def test_all_nonclaim_is_namespaced_and_does_not_load_cot(monkeypatch):
    raw = pd.DataFrame({
        "customer_id": [1, 2],
        "label": [0, 1],
        "amount": [1.0, 2.0],
    })
    monkeypatch.setattr(ml_baseline, "load_dataset", lambda config, split: raw)
    monkeypatch.setattr(ml_baseline, "add_features", lambda frame: frame)
    monkeypatch.setattr(
        ml_baseline, "build_standard_features",
        lambda frame: _frame("shared"),
    )
    monkeypatch.setattr(
        ml_baseline, "build_llm_profile_features",
        lambda frame, config: _frame("shared"),
    )
    monkeypatch.setattr(
        ml_baseline, "build_handcrafted_features",
        lambda frame, config: _frame("shared"),
    )
    monkeypatch.setattr(
        ml_baseline, "load_cot_features",
        lambda *args: (_ for _ in ()).throw(AssertionError("CoT was loaded")),
    )
    config = {
        "dataset": {"name": "gender"},
        "output": {"base_dir": "unused"},
    }
    pack = ml_baseline.build_feature_sets(config, ["all_nonclaim"])[
        "all_nonclaim"
    ]
    assert pack["columns"] == [
        "standard__shared", "profile__shared", "handcrafted__shared"
    ]


def test_rebuild_jobs_cover_all_cells_and_reuse_only_rosbank_cache(tmp_path):
    rows = build_jobs(tmp_path, "spherical_kmeans")
    assert len(rows) == 6
    assert all(row["candidates"] == list(
        BACKEND_CANDIDATES["spherical_kmeans"]
    ) for row in rows)
    cached = [row for row in rows if "embedding_cache_cell" in row]
    assert {(row["dataset"], row["model"]) for row in cached} == {
        ("rosbank", "qwen"), ("rosbank", "gpt_oss")
    }
    age_sources = {
        row["source_root"] for row in rows if row["dataset"] == "age"
    }
    assert all("guided_zero_shot_v4__age_opaque" in path for path in age_sources)


def test_quality_rebuild_reuses_selected_rosbank_sweep_cell(tmp_path):
    selection_path = tmp_path / "rosbank" / "selection.json"
    selection = {
        "selected_configuration": {
            "geometry": "centered",
            "assignment_quantile": 0.95,
            "min_client_coverage": 10,
        }
    }
    root = selected_rosbank_sweep_root(
        selection_path, selection, "qwen"
    )
    assert root == (
        selection_path.parent / "assignment" / "centered"
        / "q95_coverage_10" / "qwen"
    )
    source_cell = root / "rosbank" / "qwen" / "seed_17"
    source_cell.mkdir(parents=True)
    destination = tmp_path / "final" / "rosbank" / "qwen" / "seed_17"
    expose_cell(source_cell, destination)
    assert destination.is_symlink()
    assert destination.resolve() == source_cell.resolve()


def test_union_features_preserves_coverage_and_prefixes(tmp_path):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    output = tmp_path / "union.parquet"
    _frame("cot_0001").to_parquet(left, index=False)
    _frame("cot_0002").to_parquet(right, index=False)
    union_features(left, right, output)
    frame = pd.read_parquet(output)
    assert list(frame.columns) == [
        "customer_id", "label", "cot_qwen__0001", "cot_gpt__0002"
    ]


def test_fidelity_seed_axes_are_not_confounded():
    assert (17, 101) in SEED_PAIRS
    assert (101, 17) in SEED_PAIRS
    assert len(SEED_PAIRS) == len(set(SEED_PAIRS))


def test_semantic_union_matches_common_medoid_embeddings_and_preserves_unmatched(
    tmp_path,
):
    qwen = tmp_path / "qwen"
    gpt = tmp_path / "gpt"
    for root, names, medoids in (
        (qwen, ["cot_q1", "cot_q2"], ["cars", "food"]),
        (gpt, ["cot_g1", "cot_g2"], ["automobile", "travel"]),
    ):
        selected = root / "selected_clusters"
        selected.mkdir(parents=True)
        atomic_write_json(selected / "cluster_model.json", {
            "feature_names": names,
            "selected_feature_names": names,
            "cluster_meta": [
                {"feature": name, "medoid": medoid}
                for name, medoid in zip(names, medoids)
            ],
        })
    vectors = {
        "cars": [1.0, 0.0, 0.0],
        "automobile": [0.99, 0.01, 0.0],
        "food": [0.0, 1.0, 0.0],
        "travel": [0.0, 0.0, 1.0],
    }

    def fake_embedder(texts, model_name):
        assert model_name == "test-e5"
        return np.asarray([vectors[text] for text in texts])

    mapping = semantic_union_mapping(
        qwen,
        gpt,
        embedding_model="test-e5",
        minimum_similarity=0.50,
        embedder=fake_embedder,
    )
    assert mapping["pairs"]
    assert mapping["method"] == "mutual_nearest_common_centered_e5_medoids"

    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    output = tmp_path / "semantic.parquet"
    pd.DataFrame({
        "customer_id": [1, 2], "label": [0, 1],
        "cot_q1": [1, 0], "cot_q2": [0, 1],
    }).to_parquet(left, index=False)
    pd.DataFrame({
        "customer_id": [1, 2], "label": [0, 1],
        "cot_g1": [0, 1], "cot_g2": [1, 0],
    }).to_parquet(right, index=False)
    metadata = semantic_union_features(left, right, output, mapping)
    frame = pd.read_parquet(output)
    assert len(frame) == 2
    assert any(column.startswith("cot_shared_") for column in frame)
    assert len(metadata) == len(frame.columns) - 2


def test_ranked_occlusion_uses_probability_impact_not_column_order():
    class LinearProbability:
        classes_ = np.array([0, 1])

        def predict_proba(self, values):
            score = np.clip(
                0.5 + values[:, 0] * 0.01 + values[:, 1] * 0.20,
                0.0, 1.0,
            )
            return np.column_stack([1 - score, score])

    result = ranked_occlusion(
        LinearProbability(), np.array([1.0, 1.0]),
        ["cot_first", "cot_second"], 2, 10,
    )
    assert result["ranked_features"][0] == "cot_second"
    assert set(result["top_k"]) == {"1", "3", "5", "10"}


def test_ranked_occlusion_renormalizes_normalized_count_interventions():
    seen = []

    class RecordingProbability:
        classes_ = np.array([0, 1])

        def predict_proba(self, values):
            seen.extend(np.asarray(values, dtype=float).copy())
            score = np.clip(0.2 + values[:, 0] * 0.6, 0.0, 1.0)
            return np.column_stack([1 - score, score])

    ranked_occlusion(
        RecordingProbability(),
        np.array([0.25, 0.75]),
        ["cot_first", "cot_second"],
        2,
        2,
        encoding="normalized_count",
    )
    # The batched single-feature deletions are calls 1 and 2.  Each surviving
    # normalized-count vector must remain on the probability simplex.
    assert np.allclose(np.asarray(seen[1:3]).sum(axis=1), 1.0)


def test_bootstrap_fidelity_uses_canonical_jensen_shannon_metric():
    teacher = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    surrogate = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    result = bootstrap_fidelity(
        teacher,
        surrogate,
        np.array([0, 1, 0]),
        samples=8,
        seed=17,
    )
    assert "jensen_shannon_divergence" in result
    assert "jensen_shannon" not in result
    assert result["jensen_shannon_divergence"]["ci_low"] >= 0.0


def test_teacher_selection_reuse_requires_matching_prediction_hashes(tmp_path):
    paths = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"teacher_{split}.csv"
        path.write_text("customer_id,label\n1,0\n", encoding="utf-8")
        paths[split] = str(path)
    selection = tmp_path / "teacher_selection.json"
    atomic_write_json(selection, {
        "selected": {
            "paths": paths,
            "file_hashes": files_fingerprint(paths.values()),
        }
    })
    assert reusable_teacher_selection(selection)
    Path(paths["test"]).write_text(
        "customer_id,label\n1,1\n", encoding="utf-8"
    )
    assert not reusable_teacher_selection(selection)


def test_fidelity_v2_selects_surrogate_on_validation_and_writes_manifest(
    tmp_path, monkeypatch
):
    feature_paths = {}
    teacher_paths = {}
    for split, size in (("train", 30), ("val", 12), ("test", 12)):
        labels = np.arange(size) % 2
        signal = labels.astype(float)
        features = pd.DataFrame({
            "customer_id": np.arange(size),
            "label": labels,
            "cot_a": signal,
            "cot_b": 1.0 - signal,
            "cot_noise": np.arange(size) % 3 == 0,
        })
        teacher = pd.DataFrame({
            "customer_id": np.arange(size),
            "label": labels,
            "teacher_prob_0": 0.85 - 0.70 * signal,
            "teacher_prob_1": 0.15 + 0.70 * signal,
        })
        feature_paths[split] = tmp_path / f"features_{split}.parquet"
        teacher_paths[split] = tmp_path / f"teacher_{split}.csv"
        features.to_parquet(feature_paths[split], index=False)
        teacher.to_csv(teacher_paths[split], index=False)
    output = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "run_fidelity_analysis.py",
        "--train-features", str(feature_paths["train"]),
        "--val-features", str(feature_paths["val"]),
        "--test-features", str(feature_paths["test"]),
        "--train-teacher", str(teacher_paths["train"]),
        "--val-teacher", str(teacher_paths["val"]),
        "--test-teacher", str(teacher_paths["test"]),
        "--output-dir", str(output),
        "--feature-counts", "2",
        "--bootstrap-samples", "5",
        "--permutation-controls", "2",
        "--execute",
    ])
    fidelity_main()
    payload = json.loads(
        (output / "surrogate_selection.json").read_text(encoding="utf-8")
    )
    assert payload["selection_split"] == "validation"
    assert (output / "fidelity_stage.json").is_file()
    assert (output / "fidelity_metrics.json").is_file()

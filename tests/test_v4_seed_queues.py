import json

import numpy as np

from scripts.run_llm_seed_stability import completed_marker
from scripts.run_v4_cluster_seed_stability import (
    assignment_agreement,
    partition_for_seed,
)
from scripts.run_v4_ml_suite import completed_ml_intact


def test_minibatch_seed_partition_is_reproducible():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(120, 8)).astype(np.float32)
    first = partition_for_seed(
        values,
        candidate="k_4",
        backend="minibatch_kmeans",
        seed=101,
    )
    second = partition_for_seed(
        values,
        candidate="k_4",
        backend="minibatch_kmeans",
        seed=101,
    )
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 4


def test_assignment_agreement_aligns_by_claim_id():
    import pandas as pd

    left = pd.DataFrame({
        "claim_id": ["a", "b", "c"],
        "cluster_index": [0, 0, 1],
        "assigned": [True, True, True],
    })
    right = pd.DataFrame({
        "claim_id": ["c", "a", "b"],
        "cluster_index": [7, 2, 2],
        "assigned": [True, True, True],
    })
    metrics = assignment_agreement(left, right)
    assert metrics["ari"] == 1.0
    assert metrics["nmi"] == 1.0


def test_completion_marker_requires_exact_seed(tmp_path):
    path = tmp_path / "complete.json"
    path.write_text(json.dumps({
        "status": "completed",
        "run_id": "run",
        "dataset": "age",
        "model_slug": "qwen",
        "generation_seed": 101,
    }), encoding="utf-8")
    assert completed_marker(
        path,
        run_id="run",
        dataset="age",
        model="qwen",
        seed=101,
    )
    assert not completed_marker(
        path,
        run_id="run",
        dataset="age",
        model="qwen",
        seed=947,
    )


def test_ml_completion_rejects_modified_output(tmp_path):
    output = tmp_path / "ml_metrics.json"
    output.write_text("{}", encoding="utf-8")
    from src.experiments.artifacts import files_fingerprint

    stage_dir = tmp_path / "stages"
    stage_dir.mkdir()
    (stage_dir / "ml.json").write_text(json.dumps({
        "state": "completed",
        "outputs": [str(output)],
        "output_files": files_fingerprint([output]),
    }), encoding="utf-8")
    assert completed_ml_intact(tmp_path)
    output.write_text('{"changed": true}', encoding="utf-8")
    assert not completed_ml_intact(tmp_path)

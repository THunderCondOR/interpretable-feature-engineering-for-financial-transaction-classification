import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from scripts.cluster_stability_experiment import parse_variants
from scripts.run_fidelity_analysis import (
    load_cell,
    teacher_probabilities,
    tree_decision_paths,
)
from scripts.run_model_queue import _wait_for
from scripts.select_fidelity_teacher import evaluate_candidate

ROOT = Path(__file__).parents[1]


def test_stability_axes_vary_one_factor_at_a_time():
    args = argparse.Namespace(
        seeds=[17, 101, 947],
        distance_thresholds=[0.008, 0.010, 0.012],
        n_clusters=[100, 200, 400, 800],
        coverage_thresholds=[2, 5, 10],
        feature_encodings=["binary", "raw_count", "normalized_count"],
    )
    variants = parse_variants(args)
    by_axis = {}
    for variant in variants:
        by_axis.setdefault(variant.axis, []).append(variant)
    assert len(by_axis["clustering_order_seed"]) == 3
    assert {variant.seed for variant in by_axis["distance_threshold"]} == {17}
    assert {variant.seed for variant in by_axis["fixed_cluster_count"]} == {17}
    assert {variant.seed for variant in by_axis["min_client_coverage"]} == {17}
    assert {variant.seed for variant in by_axis["feature_encoding"]} == {17}
    assert len({variant.name for variant in variants}) == len(variants)
    configured = parse_variants(args, baseline_threshold=0.123)
    assert {
        variant.distance_threshold
        for variant in configured
        if variant.axis in {"clustering_order_seed", "min_client_coverage", "feature_encoding"}
    } == {0.123}


def test_queue_dependency_rejects_stale_run_id(tmp_path):
    dependency = tmp_path / "selection.json"
    dependency.write_text(json.dumps({"run_id": "old-run"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Stale queue dependency"):
        _wait_for(
            dependency,
            timeout=0,
            poll=0,
            expected_run_id="new-run",
        )


def test_teacher_selection_reads_filtered_combined_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = []
    for split in ("train", "val", "test"):
        for customer_id, label in ((1, 0), (2, 1)):
            rows.append(
                {
                    "feature_set": "handcrafted",
                    "classifier": "xgboost",
                    "seed": 17,
                    "split": split,
                    "customer_id": customer_id,
                    "label": label,
                    "probability_0": 0.9 if label == 0 else 0.1,
                    "probability_1": 0.1 if label == 0 else 0.9,
                }
            )
            rows.append(
                {
                    "feature_set": "standard",
                    "classifier": "xgboost",
                    "seed": 17,
                    "split": split,
                    "customer_id": customer_id,
                    "label": label,
                    "probability_0": 0.5,
                    "probability_1": 0.5,
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    metrics = evaluate_candidate(
        path,
        filters={
            "feature_set": "handcrafted",
            "classifier": "xgboost",
            "seed": 17,
        },
        split="val",
    )
    assert metrics["n"] == 2
    assert metrics["balanced_accuracy"] == 1.0


def test_fidelity_loader_filters_teacher_and_validates_probabilities(tmp_path):
    features_path = tmp_path / "features.parquet"
    teacher_path = tmp_path / "teacher.jsonl"
    pd.DataFrame(
        {
            "customer_id": [1, 2],
            "label": [0, 1],
            "cot_a": [1.0, 0.0],
        }
    ).to_parquet(features_path, index=False)
    teacher_rows = [
        {
            "customer_id": customer_id,
            "label": label,
            "split": "train",
            "feature_set": feature_set,
            "probability_0": p0,
            "probability_1": 1 - p0,
        }
        for feature_set, p0 in (("handcrafted", 0.8), ("standard", 0.6))
        for customer_id, label in ((1, 0), (2, 1))
    ]
    teacher_path.write_text(
        "".join(json.dumps(row) + "\n" for row in teacher_rows),
        encoding="utf-8",
    )
    merged = load_cell(
        features_path,
        teacher_path,
        filters={"feature_set": "handcrafted"},
        split="train",
    )
    probabilities = teacher_probabilities(merged)
    assert probabilities.shape == (2, 2)
    merged.loc[0, "probability_0"] = 2.0
    with pytest.raises(ValueError, match="Invalid teacher probability"):
        teacher_probabilities(merged)


def test_fidelity_loader_canonicalizes_numeric_string_teacher_ids(tmp_path):
    features_path = tmp_path / "features.parquet"
    teacher_path = tmp_path / "teacher.csv"
    pd.DataFrame({
        "customer_id": [1, 2],
        "label": [0, 1],
        "cot_a": [1.0, 0.0],
    }).to_parquet(features_path, index=False)
    pd.DataFrame({
        "customer_id": ["1", "2"],
        "label": [0, 1],
        "split": ["train", "train"],
        "probability_0": [0.9, 0.2],
        "probability_1": [0.1, 0.8],
    }).to_csv(teacher_path, index=False)

    merged = load_cell(features_path, teacher_path, split="train")

    assert merged["customer_id"].tolist() == [1, 2]
    assert teacher_probabilities(merged).shape == (2, 2)


def test_tree_decision_paths_include_semantic_steps_and_leaf_distribution():
    values = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.asarray([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=2, random_state=17).fit(values, labels)
    frame = pd.DataFrame(
        {
            "customer_id": [10, 11, 12, 13],
            "label": labels,
            "cot_a": values[:, 0],
            "cot_b": values[:, 1],
        }
    )
    records = tree_decision_paths(
        model,
        frame,
        ["cot_a", "cot_b"],
        ["Frequent grocery activity", "Regular mobility activity"],
    )
    assert len(records) == 4
    assert records[0]["steps"]
    assert "semantic_name" in records[0]["steps"][0]
    assert abs(sum(records[0]["leaf_probabilities"]) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "command",
    [
        [
            "scripts/prepare_grounding_sample.py",
            "--output",
            "unused-grounding.jsonl",
        ],
        [
            "scripts/summarize_grounding_judges.py",
            "--inputs",
            "unused-judge.jsonl",
            "--output-prefix",
            "unused-summary",
        ],
        [
            "scripts/compare_cot_clusters.py",
            "--left-root",
            "unused-left",
            "--right-root",
            "unused-right",
            "--output",
            "unused-cross.json",
        ],
    ],
)
def test_offline_analysis_clis_are_dry_run_by_default(command):
    result = subprocess.run(
        [sys.executable, *command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"mode": "dry-run"' in result.stdout
    for value in ("unused-grounding.jsonl", "unused-summary", "unused-cross.json"):
        assert not (ROOT / value).exists()


def test_launcher_accepts_explicit_python_and_remains_dry_run():
    result = subprocess.run(
        [
            "bash",
            "scripts/launch_model_queues.sh",
            "--run-id",
            "safe-test",
            "--qwen-config",
            "configs/v2/qwen.yaml",
            "--gpt-config",
            "configs/v2/gpt_oss.yaml",
            "--python-bin",
            sys.executable,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.count("DRY RUN: tmux new-session") == 3
    assert sys.executable in result.stdout

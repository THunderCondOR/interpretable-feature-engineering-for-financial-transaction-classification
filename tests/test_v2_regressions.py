import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_grounding_sample import _verified_evidence
from scripts.run_gender_v2 import _intervals_overlap
from src.data.loader import load_dataset
from src.experiments.artifacts import ensure_run_manifest, fingerprint
from src.experiments.config_builder import build_runtime_config
from src.pipeline.llm_eval import balanced_accuracy_interval
from src.pipeline.semantic_features import fit_semantic_space, transform_semantic_space
from src.reporting.analysis_report import load_bundle

ROOT = Path(__file__).parents[1]


def test_main_pipeline_is_dry_run_and_reports_api_stages():
    result = subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--config",
            "configs/gender.yaml",
            "--steps",
            "stats",
            "--splits",
            "test",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["mode"] == "dry-run"
    assert plan["api_steps"] == []


def test_main_pipeline_refuses_api_execution_without_until_complete():
    result = subprocess.run(
        [
            sys.executable,
            "run_pipeline.py",
            "--config",
            "configs/gender.yaml",
            "--steps",
            "cot",
            "--splits",
            "test",
            "--execute",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "API stages require --until-complete" in result.stderr


def test_runtime_config_is_versioned_and_does_not_mutate_inputs(tmp_path):
    base = {
        "dataset": {"name": "gender", "splits": {}, "columns": {}},
        "pipeline": {},
        "llm": {},
        "output": {},
    }
    profile = {
        "experiment": {"model_slug": "Qwen Test"},
        "generation": {"model": "Qwen/Test", "temperature": 0.8, "seed": 17},
        "claims_generation": {"model": "Qwen/Test", "temperature": 0.0, "seed": 17},
        "execution": {
            "initial_concurrency": 64,
            "fallback_concurrency": 10,
            "recovery_clean_batches": 10,
            "cooldown_seconds": 60,
        },
    }
    original = json.loads(json.dumps(base))
    config = build_runtime_config(
        base,
        profile,
        run_id="reviewer-v2",
        variant="neutral_robust_zero_shot",
        seed=17,
        results_root=tmp_path,
    )
    assert base == original
    assert config["output"]["base_dir"].endswith(
        "gender/neutral_robust_zero_shot/qwen_test/seed_17"
    )
    assert config["llm"]["initial_concurrency"] == 64
    assert config["llm"]["fallback_concurrency"] == 10
    assert config["pipeline"]["few_shot_per_class"] == 0


def test_client_filter_is_applied_before_column_rename(tmp_path):
    data = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "when": ["2024-01-01"] * 3,
            "value": [1.0, 2.0, 3.0],
            "category": ["a", "b", "c"],
            "target": [0, 1, 0],
        }
    ).to_csv(data, index=False)
    config = {
        "dataset": {
            "splits": {"val": str(data)},
            "client_ids_by_split": {"val": [2, 3]},
            "columns": {
                "customer_id": "id",
                "datetime": "when",
                "amount": "value",
                "category": "category",
                "label": "target",
            },
        }
    }
    loaded = load_dataset(config, "val")
    assert loaded["customer_id"].tolist() == [2, 3]

    full_context = load_dataset(
        config,
        "val",
        apply_client_filter=False,
    )
    assert full_context["customer_id"].tolist() == [1, 2, 3]


def test_manifest_invalidates_when_input_content_changes_in_place(tmp_path):
    data = tmp_path / "train.csv"
    prompt = tmp_path / "system.txt"
    data.write_text("id,label\n1,0\n", encoding="utf-8")
    prompt.write_text("first prompt", encoding="utf-8")
    config = {
        "experiment": {"run_id": "v2", "variant": "neutral", "seed": 17},
        "dataset": {"name": "gender", "splits": {"train": str(data)}},
        "prompts": {"base_dir": str(tmp_path), "system": prompt.name},
        "output": {"base_dir": str(tmp_path / "output")},
    }
    ensure_run_manifest(config, repo_root=tmp_path)
    data.write_text("id,label\n1,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible experiment"):
        ensure_run_manifest(config, repo_root=tmp_path)


def test_tfidf_is_fit_on_train_and_frozen_for_target_splits():
    config = {
        "clustering": {
            "embedding_model": "tf-idf",
            "n_clusters": 2,
            "min_client_coverage": 1,
            "max_train_distance": 1.0,
            "max_assign_distance": 1.0,
            "feature_encoding": "binary",
        },
        "feature_selection": {"enabled": False},
    }
    train = [
        {"customer_id": 1, "label": 0, "claims": ["alpha grocery"]},
        {"customer_id": 2, "label": 1, "claims": ["beta travel"]},
    ]
    model = fit_semantic_space(config, train)
    vocabulary_before = dict(model["embedding_transformer"].vocabulary_)
    target = [{"customer_id": 10, "label": 0, "claims": ["unseenword grocery"]}]
    transformed = transform_semantic_space(config, target, model)
    assert model["embedding_transformer"].vocabulary_ == vocabulary_before
    assert "unseenword" not in vocabulary_before
    assert transformed.filter(like="cot_").shape[1] == 2


def test_grounding_requires_exact_hashed_prompt_inputs():
    stats = {"client_stats": "observed stats"}
    summary = "train-only summary"
    prompt = {
        "client_stats": stats["client_stats"],
        "client_stats_hash": fingerprint(stats["client_stats"]),
        "summary_stats_hash": fingerprint(summary),
        "prompt_hash": "prompt-1",
    }
    claim = {"source_prompt_hashes": ["prompt-1"]}
    evidence, provenance = _verified_evidence(
        claim_record=claim,
        prompt_record=prompt,
        stats_record=stats,
        train_summary=summary,
        allow_unverified_legacy=False,
    )
    assert provenance == "verified_exact_prompt_inputs"
    assert evidence["client_stats"] == "observed stats"
    tampered = {**prompt, "client_stats": "different"}
    with pytest.raises(ValueError, match="provenance failed"):
        _verified_evidence(
            claim_record=claim,
            prompt_record=tampered,
            stats_record=stats,
            train_summary=summary,
            allow_unverified_legacy=False,
        )


def test_bootstrap_interval_is_deterministic_and_selection_requires_overlap():
    truth = [0, 0, 1, 1, 1, 0]
    prediction = [0, 1, 1, 1, 0, 0]
    first = balanced_accuracy_interval(truth, prediction, n_bootstrap=100, seed=137)
    second = balanced_accuracy_interval(truth, prediction, n_bootstrap=100, seed=137)
    assert first == second
    assert first["lower"] <= first["upper"]
    assert _intervals_overlap(
        {"balanced_accuracy_ci": {"lower": 0.70, "upper": 0.80}},
        {"balanced_accuracy_ci": {"lower": 0.79, "upper": 0.82}},
    )
    assert not _intervals_overlap(
        {"balanced_accuracy_ci": {"lower": 0.60, "upper": 0.70}},
        {"balanced_accuracy_ci": {"lower": 0.71, "upper": 0.80}},
    )


def test_empirical_report_loader_never_substitutes_synthetic_metrics(tmp_path):
    root = tmp_path / "gender" / "neutral" / "qwen" / "seed_17"
    root.mkdir(parents=True)
    manifest = {
        "manifest_version": 2,
        "run_id": "reviewer-v2",
        "model_id": "Qwen/Test",
        "config": {"dataset": {"name": "gender"}},
        "config_sha256": "config",
        "dataset_files": {},
        "prompt_files": {},
        "git_revision": "revision",
        "runtime": {"packages": {}},
    }
    manifest["manifest_sha256"] = fingerprint({
        "manifest_version": manifest["manifest_version"],
        "config_sha256": manifest["config_sha256"],
        "dataset_files": manifest["dataset_files"],
        "prompt_files": manifest["prompt_files"],
        "git_revision": manifest["git_revision"],
        "packages": manifest["runtime"]["packages"],
    })
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    clusters = {
        "cluster_meta": [
            {
                "cluster_id": "clu_a",
                "medoid": "client frequently uses grocery stores",
                "unique_clients": 12,
                "occurrences": 18,
            }
        ]
    }
    (root / "cot_clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    np.savez(root / "cot_cluster_model.npz", centroids=np.asarray([[1.0, 0.0]]))
    (root / "llm_metrics_test.json").write_text(
        json.dumps({"split": "test", "balanced_accuracy": 0.731}),
        encoding="utf-8",
    )
    bundle = load_bundle(tmp_path, "reviewer-v2", strict=True)
    assert bundle["synthetic"] is False
    assert bundle["metrics"][0]["balanced_accuracy"] == 0.731
    assert bundle["clusters"][0]["medoid"] == "client frequently uses grocery stores"
    assert bundle["clusters"][0]["cluster_id"] == "reviewer-v2::gender::Qwen/Test::clu_a"
    assert bundle["clusters"][0]["source_cluster_id"] == "clu_a"

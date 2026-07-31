import json
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.run_cv_lora_queue import materialize_config
from src.models.lora_trainer import _build_lora_instruction, _classification_metrics


def test_cv_lora_uses_inner_train_validation_and_outer_test(tmp_path):
    prepared = tmp_path / "prepared"
    protocol = "public_kfold5_seed100"
    root = prepared / "datafusion_education" / protocol
    fold = root / "fold_0"
    fold.mkdir(parents=True)
    events = root / "events.parquet"
    events.touch()
    (root / "benchmark_manifest.json").write_text(
        json.dumps({"events": str(events)}), encoding="utf-8"
    )
    ids = {
        "inner_train": ["a", "b"],
        "inner_validation": ["c"],
        "outer_train": ["a", "b", "c"],
        "outer_test": ["d"],
    }
    for role, values in ids.items():
        (fold / f"{role}_ids.json").write_text(
            json.dumps(values), encoding="utf-8"
        )
    (fold / "fold_manifest.json").write_text(json.dumps({
        "dataset": "datafusion_education",
        "protocol": protocol,
        "fold": 0,
        "fold_signature": "fold-signature",
        "ids": ids,
    }), encoding="utf-8")
    profile = yaml.safe_load(Path("configs/v5/lora_qwen.yaml").read_text())
    config, _ = materialize_config(
        dataset="datafusion_education",
        fold=0,
        prepared_root=prepared,
        output_root=tmp_path / "results",
        lora_profile=profile,
    )
    assert config["lora_provenance"]["split_roles"] == {
        "train": "inner_train",
        "val": "inner_validation",
        "test": "outer_test",
    }
    assert config["lora_provenance"]["counts"] == {
        "train": 2, "val": 1, "test": 1,
    }
    assert config["lora"]["models"][0]["model_name"] == "Qwen/Qwen3-8B"
    instruction = _build_lora_instruction(config)
    assert "{TASK_DESCRIPTION}" not in instruction
    assert "Final:" not in instruction
    assert "recorded higher-education qualification" in instruction


def test_binary_lora_metrics_include_positive_class_f1():
    import numpy as np

    labels = np.array([0, 1, 1, 0])
    logits = np.array([[3, 0], [0, 3], [2, 1], [3, 0]])
    metrics = _classification_metrics(labels, logits, 2, positive_label=1)
    assert metrics["f1_positive"] == 2 / 3


def test_lora_queue_finishes_all_8b_jobs_before_optional_32b():
    result = subprocess.run([
        sys.executable, "scripts/run_cv_lora_queue.py",
        "--datasets", "berka",
        "--folds", "0",
        "--models", "qwen3_8b,qwen3_32b",
    ], check=True, capture_output=True, text=True)
    jobs = json.loads(result.stdout)["jobs"]
    assert [job["models"] for job in jobs] == [["qwen3_8b"], ["qwen3_32b"]]
    assert all(job["protocol"] == "unittab_70_30_5seed" for job in jobs)

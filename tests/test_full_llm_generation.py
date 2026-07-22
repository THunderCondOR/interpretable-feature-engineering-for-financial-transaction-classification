import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_full_llm_generation import (
    LLM_STEPS,
    completion_payload,
    pipeline_command,
    validate_completion,
)
from src.experiments.config_builder import build_runtime_config


ROOT = Path(__file__).parents[1]


def _base_config(tmp_path: Path) -> dict:
    return {
        "dataset": {
            "name": "age",
            "splits": {
                "train": str(tmp_path / "train.csv"),
                "val": str(tmp_path / "val.csv"),
                "test": str(tmp_path / "test.csv"),
            },
            "columns": {},
            "label_names": {"0": "young", "1": "old"},
        },
        "pipeline": {"n_explanation_samples": 1, "n_claims_samples": 1},
        "llm": {},
        "output": {
            "summary_stats": "summary_stats.txt",
            "clients_stats": "clients_stats.jsonl",
            "prompts": "prompts.jsonl",
            "explanations": "explanations.jsonl",
            "claims": "claims.jsonl",
            "metrics": "metrics.json",
        },
    }


def _profile() -> dict:
    return {
        "experiment": {"model_slug": "Qwen Test", "seed": 999},
        "generation": {"model": "Qwen/Test", "temperature": 0.8, "seed": 999},
        "claims_generation": {
            "model": "Qwen/Test",
            "temperature": 0.0,
            "seed": 888,
        },
        "execution": {
            "initial_concurrency": 64,
            "fallback_concurrency": 10,
            "recovery_clean_batches": 10,
            "cooldown_seconds": 60,
        },
    }


def test_runtime_config_has_independent_seeds_counts_and_exact_paths(tmp_path):
    config = build_runtime_config(
        _base_config(tmp_path),
        _profile(),
        run_id="overnight",
        variant="robust_zero_shot_v2",
        sampling_seed=137,
        generation_seed=17,
        claims_seed=23,
        ml_seed=41,
        results_root=tmp_path / "results",
        expected_client_counts={"train": 24_000, "val": 3_000, "test": 3_000},
    )

    assert config["statistics"]["summary_profile"] == "robust"
    assert config["pipeline"]["few_shot_per_class"] == 0
    assert config["experiment"]["seeds"] == {
        "sampling": 137,
        "generation": 17,
        "claims": 23,
        "ml": 41,
    }
    assert config["generation"]["seed"] == 17
    assert config["claims_generation"]["seed"] == 23
    assert config["pipeline"]["few_shot_seed"] == 137
    assert config["execution"]["events_path"].endswith(
        "logs/runs/overnight/qwen_test.events.jsonl"
    )
    assert config["llm"]["events_path"] == config["execution"]["events_path"]
    assert config["evaluation"]["ml_seed"] == 41
    assert config["output"]["base_dir"].endswith(
        "age/robust_zero_shot_v2/qwen_test/seed_17"
    )
    assert config["dataset"]["input_paths_by_split"]["train"].endswith("train.csv")
    assert config["dataset"]["expected_client_counts"]["train"] == 24_000
    assert config["output"]["paths_by_split"]["test"]["claims"].endswith(
        "claims_test.jsonl"
    )


def test_full_runner_is_dry_run_and_does_not_materialize_files(tmp_path):
    runtime_config = tmp_path / "runtime.yaml"
    completion_marker = tmp_path / "completed.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_full_llm_generation.py",
            "--dataset",
            "rosbank",
            "--model-config",
            "configs/v2/qwen.yaml",
            "--runtime-config",
            str(runtime_config),
            "--completion-marker",
            str(completion_marker),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["mode"] == "dry-run"
    assert plan["steps"] == list(LLM_STEPS)
    assert plan["splits"] == ["train", "val", "test"]
    assert plan["expected_counts"] == {"train": 4_000, "val": 500, "test": 500}
    assert not runtime_config.exists()
    assert not completion_marker.exists()


def test_full_runner_requires_all_api_guards_before_any_write(tmp_path):
    runtime_config = tmp_path / "runtime.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_full_llm_generation.py",
            "--dataset",
            "age",
            "--model-config",
            "configs/v2/qwen.yaml",
            "--runtime-config",
            str(runtime_config),
            "--execute",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--execute-api" in result.stderr
    assert not runtime_config.exists()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _complete_fixture(tmp_path: Path) -> tuple[dict, dict[str, set[int]]]:
    output = tmp_path / "output"
    config = {
        "experiment": {
            "run_id": "night",
            "model_slug": "qwen",
            "variant": "robust_zero_shot_v2",
        },
        "dataset": {
            "name": "age",
            "label_names": {"0": "young", "1": "old"},
            "expected_client_counts": {"test": 2},
        },
        "output": {
            "base_dir": str(output),
            "clients_stats": "clients_stats.jsonl",
            "prompts": "prompts.jsonl",
            "explanations": "explanations.jsonl",
            "claims": "claims.jsonl",
            "paths_by_split": {},
        },
    }
    _write_jsonl(
        output / "clients_stats_test.jsonl",
        [{"customer_id": 1}, {"customer_id": 2}],
    )
    prompts = [
        {"customer_id": 1, "prompt_hash": "prompt-1"},
        {"customer_id": 2, "prompt_hash": "prompt-2"},
    ]
    _write_jsonl(output / "prompts_test.jsonl", prompts)
    _write_jsonl(
        output / "explanations_test.jsonl",
        [
            {
                "customer_id": row["customer_id"],
                "explanation": "A complete behavioral explanation.",
                "predicted": row["customer_id"] - 1,
                "error": None,
                "error_type": None,
                "generation_signature": f"generation-{row['customer_id']}",
                "prompt_hash": row["prompt_hash"],
            }
            for row in prompts
        ],
    )
    _write_jsonl(
        output / "claims_test.jsonl",
        [
            {
                "customer_id": row["customer_id"],
                "claims": ["An atomic behavioral claim."],
                "error": None,
                "error_type": None,
                "generation_signature": f"claims-{row['customer_id']}",
                "source_explanation_hash": f"source-{row['customer_id']}",
                "source_prompt_hashes": [row["prompt_hash"]],
            }
            for row in prompts
        ],
    )
    (output / "llm_metrics_test.json").write_text(
        json.dumps(
            {
                "split": "test",
                "n_rows": 2,
                "n_scored": 2,
                "n_skipped": 0,
                "n_errors": 0,
                "coverage": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps({"manifest_sha256": "manifest-identity"}),
        encoding="utf-8",
    )
    return config, {"test": {1, 2}}


def test_completion_evidence_requires_exact_successful_outputs(tmp_path):
    config, expected = _complete_fixture(tmp_path)
    evidence = validate_completion(config, ["test"], expected)
    marker = completion_payload(config, ["test"], evidence)
    assert marker["status"] == "completed"
    assert marker["manifest_sha256"] == "manifest-identity"
    assert marker["expected_counts"] == {"test": 2}
    assert marker["completion_signature"]


def test_completion_rejects_duplicate_clients(tmp_path):
    config, expected = _complete_fixture(tmp_path)
    claims_path = Path(config["output"]["base_dir"]) / "claims_test.jsonl"
    first = claims_path.read_text(encoding="utf-8").splitlines()[0]
    with claims_path.open("a", encoding="utf-8") as file:
        file.write(first + "\n")
    with pytest.raises(ValueError, match="Duplicate customer_id=1"):
        validate_completion(config, ["test"], expected)


def test_pipeline_command_contains_only_reusable_llm_stages():
    command = pipeline_command(Path("generated.yaml"), ["train", "test"])
    assert command[command.index("--steps") + 1] == "stats,prompts,cot,llm_eval,claims"
    assert command[command.index("--splits") + 1] == "train,test"
    assert "cot_features" not in command
    assert "ml" not in command

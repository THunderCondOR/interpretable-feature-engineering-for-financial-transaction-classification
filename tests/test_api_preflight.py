from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.api_preflight import (
    PreflightError,
    _required_environment,
    probe_models,
    run_preflight,
    validate_datasets,
    validate_git_clean,
)


ROOT = Path(__file__).parents[1]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture_configs(tmp_path: Path) -> tuple[list[Path], dict[str, dict[str, int]]]:
    expected: dict[str, dict[str, int]] = {}
    configs: list[Path] = []
    next_id = 0
    for dataset in ("gender", "age", "rosbank"):
        split_paths: dict[str, str] = {}
        expected[dataset] = {}
        for split, count in (("train", 3), ("val", 2), ("test", 1)):
            ids = list(range(next_id, next_id + count))
            next_id += count
            path = tmp_path / "data" / dataset / f"{split}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"cid": ids, "amount": [1.0] * count}).to_csv(path, index=False)
            split_paths[split] = str(path)
            expected[dataset][split] = count

        prompt_dir = tmp_path / "prompts" / dataset
        prompt_dir.mkdir(parents=True)
        prompt_names = {}
        for key in ("system", "user", "claims_system", "claims_user"):
            prompt = prompt_dir / f"{key}.txt"
            prompt.write_text(key, encoding="utf-8")
            prompt_names[key] = prompt.name
        configs.append(
            _write_json(
                tmp_path / "configs" / f"{dataset}.yaml",
                {
                    "dataset": {
                        "name": dataset,
                        "columns": {"customer_id": "cid"},
                        "splits": split_paths,
                    },
                    "prompts": {"base_dir": str(prompt_dir), **prompt_names},
                },
            )
        )
    return configs, expected


def _model_profiles(tmp_path: Path) -> tuple[Path, Path]:
    paths = []
    for slug, model in (("qwen", "Qwen/test"), ("gpt_oss", "Openai/test")):
        paths.append(
            _write_json(
                tmp_path / f"{slug}.yaml",
                {
                    "experiment": {"model_slug": slug},
                    "generation": {"model": model},
                    "claims_generation": {"model": model},
                },
            )
        )
    return paths[0], paths[1]


def test_environment_rejects_empty_and_unresolved_credentials():
    with pytest.raises(PreflightError, match="API_KEY"):
        _required_environment({"API_BASE_URL": "https://example.test", "API_KEY": ""})
    with pytest.raises(PreflightError, match="unresolved"):
        _required_environment(
            {"API_BASE_URL": "https://example.test", "API_KEY": "${API_KEY}"}
        )


def test_dataset_preflight_checks_counts_prompts_and_split_leakage(tmp_path):
    configs, expected = _fixture_configs(tmp_path)
    assert validate_datasets(tmp_path, configs, expected) == expected

    leaked = pd.read_csv(tmp_path / "data" / "gender" / "val.csv")
    leaked.loc[0, "cid"] = 0
    leaked.to_csv(tmp_path / "data" / "gender" / "val.csv", index=False)
    expected["gender"]["val"] = len(set(leaked["cid"]))
    with pytest.raises(PreflightError, match="Client leakage"):
        validate_datasets(tmp_path, configs, expected)


def test_preflight_can_be_tested_without_git_or_network(tmp_path):
    configs, expected = _fixture_configs(tmp_path)
    qwen, gpt = _model_profiles(tmp_path)
    result = run_preflight(
        repo_root=tmp_path,
        qwen_config=qwen,
        gpt_config=gpt,
        dataset_configs=configs,
        environ={"API_BASE_URL": "https://example.test/v1", "API_KEY": "secret"},
        expected_counts=expected,
        check_git=False,
        probe=False,
    )
    assert result["clients_per_model"] == 18
    assert result["estimated_api_requests"] == 174_800
    assert result["probed_models"] == []


def test_probe_sends_one_minimal_request_per_model_without_leaking_key():
    calls: list[dict] = []
    factory_kwargs: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[])

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())
            self.closed = False

        def close(self):
            self.closed = True

    fake_client = FakeClient()

    def factory(**kwargs):
        factory_kwargs.update(kwargs)
        return fake_client

    completed = probe_models(
        [
            {"slug": "qwen", "generation_model": "Qwen/test"},
            {"slug": "gpt_oss", "generation_model": "Openai/test"},
        ],
        base_url="https://example.test/v1",
        api_key="top-secret",
        client_factory=factory,
    )
    assert completed == ["qwen", "gpt_oss"]
    assert [call["model"] for call in calls] == ["Qwen/test", "Openai/test"]
    assert all(call["max_tokens"] == 8 for call in calls)
    assert factory_kwargs == {
        "base_url": "https://example.test/v1",
        "api_key": "top-secret",
    }
    assert fake_client.closed


def test_git_clean_guard_detects_tracked_and_untracked_changes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    assert len(validate_git_clean(tmp_path)) == 40

    (tmp_path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(PreflightError, match="dirty worktree"):
        validate_git_clean(tmp_path)


def test_launcher_dry_run_has_no_writes_and_prints_guards_and_scope():
    run_id = f"dry-{uuid.uuid4().hex}"
    output_dir = ROOT / "logs" / "runs" / run_id
    result = subprocess.run(
        [
            "bash",
            "scripts/launch_model_queues.sh",
            "--run-id",
            run_id,
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
        env=os.environ.copy(),
    )
    assert "estimated API requests: 174,800" in result.stdout
    assert result.stdout.count("DRY RUN: tmux new-session") == 3
    assert result.stdout.count("--execute-api") == 2
    assert "bash\\ -o\\ pipefail" in result.stdout
    assert not output_dir.exists()


def test_launcher_execute_requires_all_guards_and_explicit_python():
    base = [
        "bash",
        "scripts/launch_model_queues.sh",
        "--run-id",
        "guard-test",
        "--qwen-config",
        "configs/v2/qwen.yaml",
        "--gpt-config",
        "configs/v2/gpt_oss.yaml",
        "--execute",
    ]
    missing_guards = subprocess.run(base, cwd=ROOT, capture_output=True, text=True)
    assert missing_guards.returncode == 2
    assert "requires --execute-api and --until-complete" in missing_guards.stderr

    missing_python = subprocess.run(
        [*base, "--execute-api", "--until-complete"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert missing_python.returncode == 2
    assert "explicit --python-bin" in missing_python.stderr

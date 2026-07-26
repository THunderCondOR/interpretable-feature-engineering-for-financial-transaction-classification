import json

import pytest

import src.experiments.artifacts as artifacts


def test_scheduler_resume_allows_only_git_revision_change(
    tmp_path, monkeypatch
):
    output = tmp_path / "run"
    config = {
        "experiment": {"run_id": "resume"},
        "dataset": {"name": "fixture", "splits": {}},
        "prompts": {},
        "pipeline": {},
        "output": {"base_dir": str(output)},
    }
    monkeypatch.setattr(artifacts, "git_revision", lambda _root: "old")
    path = artifacts.ensure_run_manifest(config, repo_root=tmp_path)
    original = json.loads(path.read_text())

    monkeypatch.setattr(artifacts, "git_revision", lambda _root: "new")
    with pytest.raises(RuntimeError):
        artifacts.ensure_run_manifest(config, repo_root=tmp_path)

    monkeypatch.setenv("ALLOW_SCHEDULER_CODE_RESUME", "1")
    assert artifacts.ensure_run_manifest(config, repo_root=tmp_path) == path
    assert json.loads(path.read_text()) == original


def test_scheduler_resume_rejects_config_change(tmp_path, monkeypatch):
    output = tmp_path / "run"
    config = {
        "experiment": {"run_id": "resume"},
        "dataset": {"name": "fixture", "splits": {}},
        "prompts": {},
        "pipeline": {},
        "output": {"base_dir": str(output)},
    }
    monkeypatch.setattr(artifacts, "git_revision", lambda _root: "old")
    artifacts.ensure_run_manifest(config, repo_root=tmp_path)
    monkeypatch.setattr(artifacts, "git_revision", lambda _root: "new")
    monkeypatch.setenv("ALLOW_SCHEDULER_CODE_RESUME", "1")
    changed = {
        **config,
        "pipeline": {"generation_seed": 947},
    }
    with pytest.raises(RuntimeError):
        artifacts.ensure_run_manifest(changed, repo_root=tmp_path)

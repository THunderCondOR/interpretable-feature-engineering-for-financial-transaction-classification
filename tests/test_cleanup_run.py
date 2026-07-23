import json
from pathlib import Path

import pytest

from scripts.cleanup_run import matching_result_dirs, remove_run


def _manifest(path: Path, run_id: str) -> None:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"run_id": run_id}), encoding="utf-8"
    )


def test_cleanup_matches_only_exact_manifest_run_id(tmp_path):
    results = tmp_path / "results"
    target = results / "gender" / "target"
    other = results / "gender" / "target-suffix"
    legacy = results / "legacy"
    _manifest(target, "night-run")
    _manifest(other, "night-run-2")
    legacy.mkdir(parents=True)
    (legacy / "artifact.txt").write_text("keep", encoding="utf-8")

    assert matching_result_dirs(results, "night-run") == [target]
    payload = remove_run(
        "night-run",
        results_root=results,
        logs_root=tmp_path / "logs",
        reports_root=tmp_path / "reports",
        execute=True,
        proc_root=tmp_path / "no-proc",
    )
    assert str(target.resolve()) in payload["removed"]
    assert not target.exists()
    assert other.exists()
    assert legacy.exists()


def test_cleanup_is_dry_run_by_default_and_includes_exact_logs(tmp_path):
    results, logs, reports = (
        tmp_path / "results",
        tmp_path / "logs",
        tmp_path / "reports",
    )
    target = results / "cell"
    _manifest(target, "night-run")
    (logs / "night-run").mkdir(parents=True)
    (reports / "night-run").mkdir(parents=True)
    payload = remove_run(
        "night-run",
        results_root=results,
        logs_root=logs,
        reports_root=reports,
        proc_root=tmp_path / "no-proc",
    )
    assert payload["mode"] == "dry-run"
    assert target.exists()
    assert logs.joinpath("night-run").exists()
    assert reports.joinpath("night-run").exists()


def test_cleanup_refuses_live_process(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.cleanup_run.live_processes",
        lambda *_args, **_kwargs: [{"pid": 123, "argv": ["worker", "night-run"]}],
    )
    with pytest.raises(RuntimeError, match="still has 1 live"):
        remove_run(
            "night-run",
            results_root=tmp_path / "results",
            logs_root=tmp_path / "logs",
            reports_root=tmp_path / "reports",
            execute=True,
        )

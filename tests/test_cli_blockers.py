import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_grounding_suite, run_robustness_suite


ROOT = Path(__file__).parents[1]


def test_compare_clusters_dry_run_does_not_touch_inputs_or_output(tmp_path):
    output = tmp_path / "comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_cot_clusters.py",
            "--left-root",
            str(tmp_path / "missing-left"),
            "--right-root",
            str(tmp_path / "missing-right"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"mode": "dry-run"' in result.stdout
    assert not output.exists()


def test_grounding_suite_executes_sample_preparation(monkeypatch, tmp_path):
    calls = []
    sample = tmp_path / "grounding_samples.jsonl"
    def fake_run(command, check):
        calls.append(command)
        if command[1] == "scripts/prepare_grounding_sample.py":
            sample.write_text("", encoding="utf-8")
    monkeypatch.setattr(run_grounding_suite.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_grounding_suite.py",
            "--execute",
            "--datasets",
            "gender",
            "--sources-config",
            str(tmp_path / "sources.yaml"),
            "--output-dir",
            str(tmp_path),
        ],
    )

    run_grounding_suite.main()

    assert len(calls) == 2
    assert calls[0][1] == "scripts/prepare_grounding_sample.py"
    assert "--execute" in calls[0]
    assert calls[1][1] == "scripts/build_grounding_annotation.py"
    assert (tmp_path / "grounding.commands.json").exists()


def test_grounding_cost_guard_counts_both_fresh_judges(tmp_path):
    sample = tmp_path / "sample.jsonl"
    row = {
        "sample_id": "s", "client_stats": "x" * 1000,
        "train_reference_summary": "summary", "field_semantics": "amount",
        "claim": "The client is active.",
    }
    import json
    sample.write_text(json.dumps(row) + "\n", encoding="utf-8")
    estimate = run_grounding_suite.estimate_cost(
        sample, run_grounding_suite.DEFAULT_JUDGES, 192,
    )
    assert set(estimate["models"]) == set(run_grounding_suite.DEFAULT_JUDGES)
    assert estimate["estimated_total_usd"] > 0


def test_robustness_api_guard_precedes_side_effects(monkeypatch, tmp_path):
    generated = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_robustness_suite.py",
            "--execute",
            "--execute-api",
            "--until-complete",
            "--generated-dir",
            str(generated),
        ],
    )

    with pytest.raises(RuntimeError, match="not implemented"):
        run_robustness_suite.main()

    assert not generated.exists()


def test_robustness_offline_execute_runs_materialized_queue(monkeypatch, tmp_path):
    calls = []
    generated = tmp_path / "generated"
    output = generated / "queue.json"
    config = generated / "gender.yaml"
    monkeypatch.setattr(
        run_robustness_suite,
        "materialize_offline_config",
        lambda **kwargs: config,
    )
    monkeypatch.setattr(
        run_robustness_suite,
        "commands_for",
        lambda *args, **kwargs: [["offline-command"]],
    )
    monkeypatch.setattr(
        run_robustness_suite.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_robustness_suite.py",
            "--execute",
            "--datasets",
            "gender",
            "--generated-dir",
            str(generated),
            "--output",
            str(output),
        ],
    )

    run_robustness_suite.main()

    assert calls == [["offline-command"]]
    assert output.exists()
    assert output.with_suffix(".yaml").exists()

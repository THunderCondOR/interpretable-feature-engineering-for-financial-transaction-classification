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
    monkeypatch.setattr(
        run_grounding_suite.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_grounding_suite.py",
            "--execute",
            "--datasets",
            "gender",
            "--run-roots",
            "results",
            "--run-names",
            "qwen",
            "--output-dir",
            str(tmp_path),
        ],
    )

    run_grounding_suite.main()

    assert len(calls) == 1
    assert calls[0][1] == "scripts/prepare_grounding_sample.py"
    assert "--execute" in calls[0]
    assert (tmp_path / "grounding.commands.json").exists()


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

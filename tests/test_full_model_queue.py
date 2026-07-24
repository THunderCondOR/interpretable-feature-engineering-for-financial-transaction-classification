import json
from pathlib import Path

import yaml

from scripts.run_model_queue import (
    _event,
    _job_signature,
    _output_evidence,
    default_jobs,
)
from scripts.pipeline_status import normalize_event


def _profile(model_slug: str) -> dict:
    return {"experiment": {"model_slug": model_slug}}


def test_qwen_queue_covers_all_datasets_and_excludes_offline_jobs():
    jobs = default_jobs(
        _profile("qwen"),
        model_config=Path("configs/v2/qwen.yaml"),
        run_id="nightly",
    )
    assert [job["id"] for job in jobs] == [
        "rosbank_pilot",
        "rosbank_select",
        "gender_pilot",
        "gender_select",
        "age_pilot",
        "age_select",
        "gender_qwen_full",
        "rosbank_qwen_full",
        "age_qwen_full",
    ]
    assert {job["dataset"] for job in jobs} == {"gender", "rosbank", "age"}
    assert all(job["stage"] != "robustness" for job in jobs)
    assert jobs[1]["wait_for"].endswith("qwen_rosbank_pilot.json")
    full_commands = [job["command"] for job in jobs if job["stage"] == "full"]
    assert all("--execute-api" in command for command in full_commands)
    assert all("--selected-config" in command for command in full_commands)


def test_gpt_queue_runs_rosbank_then_selected_gender_then_age():
    jobs = default_jobs(
        _profile("gpt_oss"),
        model_config=Path("configs/v2/gpt_oss.yaml"),
        run_id="nightly",
    )
    assert [job["dataset"] for job in jobs] == ["rosbank", "gender", "age"]
    assert jobs[0]["wait_for"].endswith("rosbank_selection.json")
    assert jobs[1]["wait_for"].endswith("gender_selection.json")
    assert jobs[2]["wait_for"].endswith("age_selection.json")
    assert "scripts/run_full_llm_generation.py" in jobs[0]["command"]
    assert "scripts/run_full_llm_generation.py" in jobs[1]["command"]


def test_queue_uses_signed_selected_config_counts_when_available(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    generated = Path("logs/runs/nightly/generated")
    generated.mkdir(parents=True)
    (generated / "age_selected_gpt_oss.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "expected_client_counts": {
                        "train": 8000,
                        "val": 1000,
                        "test": 3000,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    jobs = default_jobs(
        _profile("gpt_oss"),
        model_config=Path("configs/v2/gpt_oss.yaml"),
        run_id="nightly",
    )

    assert jobs[-1]["expected_client_counts"] == {
        "train": 8000,
        "val": 1000,
        "test": 3000,
    }


def test_completion_evidence_must_match_run_model_dataset(tmp_path):
    marker = tmp_path / "qwen_age.json"
    job = {"dataset": "age", "expected_outputs": [str(marker)]}
    marker.write_text(
        json.dumps({
            "run_id": "nightly",
            "dataset": "age",
            "model_slug": "qwen",
            "status": "completed",
        }),
        encoding="utf-8",
    )
    assert _output_evidence(
        job,
        expected_run_id="nightly",
        expected_model="qwen",
    )
    assert _output_evidence(
        job,
        expected_run_id="other",
        expected_model="qwen",
    ) is None


def test_job_signature_does_not_change_when_output_is_created(tmp_path):
    config = tmp_path / "profile.yaml"
    config.write_text("experiment:\n  model_slug: qwen\n", encoding="utf-8")
    marker = tmp_path / "complete.json"
    job = {
        "command": ["runner", "--config", str(config)],
        "expected_outputs": [str(marker)],
    }
    before = _job_signature(job, config)
    marker.write_text("{}", encoding="utf-8")
    after = _job_signature(job, config)
    assert before == after


def test_status_scopes_api_stages_by_split_and_keeps_queue_failures_visible():
    explanation = normalize_event({"stage": "explanations", "split": "train"})
    failure = normalize_event({"stage": "full", "event": "job_failed"})
    assert explanation["stage"] == "train_explanations"
    assert failure["stage"] == "queue"


def test_dependency_event_can_record_path_without_argument_collision(tmp_path):
    events = tmp_path / "events.jsonl"
    _event(events, event="dependency_wait", path="selection.json")
    assert json.loads(events.read_text())["path"] == "selection.json"

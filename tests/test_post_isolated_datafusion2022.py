import json

from scripts.run_post_isolated_datafusion2022 import (
    ISOLATED_DATASETS,
    resume_command,
    valid_isolated_completion,
)
from src.experiments.artifacts import fingerprint


def test_isolated_completion_requires_both_datasets_and_valid_signature(tmp_path):
    path = tmp_path / "completion.json"
    payload = {
        "status": "completed",
        "run_id": "isolated",
        "model": "qwen",
        "datasets": ISOLATED_DATASETS,
    }
    payload["completion_signature"] = fingerprint(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert valid_isolated_completion(path, run_id="isolated", model="qwen")
    assert not valid_isolated_completion(path, run_id="isolated", model="gpt_oss")
    payload["datasets"] = list(reversed(ISOLATED_DATASETS))
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not valid_isolated_completion(path, run_id="isolated", model="qwen")


def test_resume_command_targets_only_datafusion2022():
    command = resume_command(
        python_bin="/env/python", model="gpt_oss", resume_run_id="old-run"
    )
    assert command[-2:] == ["--datasets", "datafusion_education"]
    assert command[command.index("--run-id") + 1] == "old-run"
    assert command[command.index("--model") + 1] == "gpt_oss"

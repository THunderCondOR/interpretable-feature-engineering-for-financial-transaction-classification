import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.run_gender_v2 import stratified_pilot_ids
from scripts.pipeline_status import render

ROOT = Path(__file__).parents[1]


def test_gender_pilot_sampler_is_deterministic_and_has_requested_size():
    rows = []
    for cid in range(600):
        for txn in range(1 + cid % 7):
            rows.append({"customer_id": cid, "label": cid % 2, "amount": float((cid % 11 + 1) * (txn + 1))})
    frame = pd.DataFrame(rows)
    first = stratified_pilot_ids(frame)
    second = stratified_pilot_ids(frame)
    assert first == second
    assert len(first) == 400
    labels = frame.drop_duplicates("customer_id").set_index("customer_id")["label"]
    assert abs(sum(labels.loc[first]) - 200) < 15


def test_all_new_computational_runners_are_dry_run_by_default():
    commands = [
        [sys.executable, "scripts/run_gender_v2.py"],
        [sys.executable, "scripts/run_robustness_suite.py"],
        [sys.executable, "scripts/run_grounding_suite.py"],
        [sys.executable, "scripts/run_model_queue.py", "--run-id", "test", "--model-config", "configs/v2/qwen.yaml"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        assert '"mode": "dry-run"' in result.stdout


def test_tmux_launcher_dry_run_does_not_create_sessions():
    result = subprocess.run([
        "bash", "scripts/launch_model_queues.sh", "--run-id", "reviewer-v2",
        "--qwen-config", "configs/v2/qwen.yaml", "--gpt-config", "configs/v2/gpt_oss.yaml",
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    assert result.stdout.count("DRY RUN: tmux new-session") == 3


def test_pipeline_status_builds_html_from_synthetic_events(tmp_path):
    event_dir = tmp_path / "logs" / "demo"
    event_dir.mkdir(parents=True)
    (event_dir / "qwen.events.jsonl").write_text(json.dumps({
        "timestamp": 1, "dataset": "gender", "model": "qwen", "stage": "explanations",
        "event": "window_committed", "completed": 64, "expected": 128, "concurrency": 64,
    }) + "\n", encoding="utf-8")
    output = tmp_path / "status.html"
    render("demo", tmp_path / "logs", tmp_path / "results", output)
    text = output.read_text(encoding="utf-8")
    assert "gender/qwen" in text
    assert "window_committed 64/128" in text

#!/usr/bin/env python3
"""Give completed DataFusion API cells GPU priority without idling for them."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cv_llm_queue import valid_dataset_completion
from src.experiments.artifacts import atomic_write_json


DATAFUSION_ROOT = Path("results/v5/derived/cv_main_e5_public")
MODELS = ("qwen", "gpt_oss")


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def stop_lora(session: str, timeout: float = 90.0) -> bool:
    """Stop LoRA only when DataFusion has concrete work ready for the GPU."""
    if not session_exists(session):
        return False
    subprocess.run(["tmux", "send-keys", "-t", f"{session}:0.0", "C-c"], check=True)
    deadline = time.monotonic() + timeout
    while session_exists(session) and time.monotonic() < deadline:
        time.sleep(2.0)
    if session_exists(session):
        raise RuntimeError(f"LoRA session did not stop after SIGINT: {session}")
    return True


def launch_lora(run_id: str) -> None:
    session = run_id.replace("-", "_").replace(".", "_")
    if session_exists(session):
        return
    subprocess.run([
        "bash", "scripts/launch_all_lora_queue.sh",
        "--run-id", run_id, "--execute",
    ], cwd=REPO_ROOT, check=True)


def completion_ready(run_id: str, model: str) -> bool:
    marker = (
        Path("logs/runs") / run_id / "completion"
        / f"{model}_datafusion_education_dataset.json"
    )
    return valid_dataset_completion(
        marker,
        run_id=run_id,
        dataset="datafusion_education",
        model=model,
    )


def run_python(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datafusion-run-id", default="reviewer-v10-datafusion-public")
    parser.add_argument("--stability-session", default="reviewer_v10_stability")
    parser.add_argument("--lora-run-id", default="reviewer-v10-lora-all-datasets")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--state", type=Path,
        default=Path("results/v5/derived/reviewer-v10-datafusion-gpu-priority.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "wait_for": {model: f"DataFusion completion for {model}" for model in MODELS},
        "gpu_policy": "wait current stability; preempt LoRA; run DataFusion; resume LoRA",
        "derived_root": str(DATAFUSION_ROOT),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return

    previous = json.loads(args.state.read_text()) if args.state.is_file() else {}
    completed = list(previous.get("completed", []))
    state = {
        **plan,
        "state": "waiting_api",
        "completed": completed,
        "started_at": previous.get("started_at", datetime.now(timezone.utc).isoformat()),
    }
    atomic_write_json(args.state, state)
    lora_session = args.lora_run_id.replace("-", "_").replace(".", "_")

    remaining = {
        model for model in MODELS if f"offline_{model}" not in completed
    }
    while remaining:
        ready = [
            model for model in MODELS
            if model in remaining and completion_ready(args.datafusion_run_id, model)
        ]
        if not ready:
            time.sleep(max(5.0, args.poll_seconds))
            continue
        # Current stability is never preempted. LoRA is lower priority and is
        # paused only after an API completion marker makes useful work runnable.
        while session_exists(args.stability_session):
            time.sleep(max(5.0, args.poll_seconds))
        stop_lora(lora_session)
        for model in ready:
            key = f"offline_{model}"
            state["state"] = "preempting_lora"
            state["current"] = key
            atomic_write_json(args.state, state)
            run_python([
                "scripts/run_cv_offline_pipeline.py",
                "--datasets", "datafusion_education",
                "--models", model,
                "--folds", "0,1,2,3,4",
                "--run-id", args.datafusion_run_id,
                "--derived-root", str(DATAFUSION_ROOT),
                "--embedding-model", "intfloat/multilingual-e5-large",
                "--execute",
            ])
            completed.append(key)
            remaining.remove(model)
            state["completed"] = completed
            atomic_write_json(args.state, state)
        launch_lora(args.lora_run_id)

    if "datafusion_cluster_stability" not in completed:
        while session_exists(args.stability_session):
            time.sleep(max(5.0, args.poll_seconds))
        stop_lora(lora_session)
        state["state"] = "running_datafusion_stability"
        state["current"] = "datafusion_cluster_stability"
        atomic_write_json(args.state, state)
        run_python([
            "scripts/run_cv_cluster_stability.py",
            "--dataset", "datafusion_education",
            "--run-id", args.datafusion_run_id,
            "--derived-root", str(DATAFUSION_ROOT),
            "--execute",
        ])
        completed.append("datafusion_cluster_stability")
        state["completed"] = completed
        atomic_write_json(args.state, state)
        launch_lora(args.lora_run_id)

    state["state"] = "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(args.state, state)


if __name__ == "__main__":
    main()

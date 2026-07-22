"""Durable per-model queue; dry-run unless both execution guards are set."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json, files_fingerprint, fingerprint


def _gender_command(
    *,
    run_id: str,
    model_config: Path,
    stage: str,
    api: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_gender_v2.py",
        "--run-id",
        run_id,
        "--stage",
        stage,
        "--model-config",
        str(model_config),
        "--execute",
    ]
    if api:
        command.extend(["--execute-api", "--until-complete"])
    return command


def default_jobs(
    profile: dict[str, Any],
    *,
    model_config: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    model = profile["experiment"]["model_slug"]
    selection = str(Path("logs/runs") / run_id / "generated" / "gender_selection.json")
    if model == "qwen":
        return [
            {
                "id": "gender_pilot",
                "dataset": "gender",
                "stage": "pilot",
                "api": True,
                "command": _gender_command(
                    run_id=run_id,
                    model_config=model_config,
                    stage="pilot",
                    api=True,
                ),
            },
            {
                "id": "gender_select",
                "dataset": "gender",
                "stage": "select",
                "api": False,
                "expected_outputs": [selection],
                "command": _gender_command(
                    run_id=run_id,
                    model_config=model_config,
                    stage="select",
                    api=False,
                ),
            },
            {
                "id": "gender_full",
                "dataset": "gender",
                "stage": "full",
                "api": True,
                "wait_for": selection,
                "command": _gender_command(
                    run_id=run_id,
                    model_config=model_config,
                    stage="full",
                    api=True,
                ),
            },
            {
                "id": "offline_robustness",
                "dataset": "all",
                "stage": "robustness",
                "api": False,
                "command": [
                    sys.executable,
                    "scripts/run_robustness_suite.py",
                    "--run-id",
                    run_id,
                    "--execute",
                ],
            },
        ]
    if model == "gpt_oss":
        return [
            {
                "id": "gender_gpt_selected",
                "dataset": "gender",
                "stage": "selected_test_and_claims",
                "api": True,
                "wait_for": selection,
                "command": _gender_command(
                    run_id=run_id,
                    model_config=model_config,
                    stage="gpt",
                    api=True,
                ),
            }
        ]
    raise ValueError(f"Unsupported model_slug: {model}")

def _job_signature(job: dict[str, Any], model_config: Path) -> str:
    paths = [model_config]
    for value in job["command"]:
        candidate = Path(str(value))
        if candidate.is_file():
            paths.append(candidate)
    dependency = job.get("wait_for")
    if dependency and Path(dependency).is_file():
        paths.append(Path(dependency))
    return fingerprint(
        {
            "command": job["command"],
            "wait_for": dependency,
            "input_files": files_fingerprint(paths),
            "output_files": files_fingerprint(job.get("expected_outputs", [])),
        }
    )


def _output_evidence(job: dict[str, Any]) -> dict[str, Any] | None:
    outputs = [Path(value) for value in job.get("expected_outputs", [])]
    if not outputs or any(not path.is_file() for path in outputs):
        return None
    return files_fingerprint(outputs)

def _event(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps({"timestamp": time.time(), **payload}) + "\n")


def _wait_for(
    path: Path,
    *,
    timeout: float,
    poll: float,
    expected_run_id: str | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Queue dependency did not appear: {path}")
        time.sleep(poll)
    if expected_run_id:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_id") != expected_run_id:
            raise RuntimeError(
                f"Stale queue dependency {path}: run_id={payload.get('run_id')!r}, "
                f"expected {expected_run_id!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--dependency-timeout-seconds", type=float, default=86400)
    parser.add_argument("--dependency-poll-seconds", type=float, default=10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    profile = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    profile["experiment"]["run_id"] = args.run_id
    jobs = (
        json.loads(args.queue.read_text(encoding="utf-8"))
        if args.queue
        else default_jobs(
            profile,
            model_config=args.model_config,
            run_id=args.run_id,
        )
    )
    identifiers = [str(job.get("id", "")) for job in jobs]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Every queue job must have a unique non-empty id")
    if any(not isinstance(job.get("command"), list) for job in jobs):
        raise ValueError("Every queue job must contain an argv-list command")

    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "profile": str(args.model_config),
        "jobs": jobs,
    }, indent=2))
    if not args.execute:
        return
    if not args.until_complete:
        raise ValueError("--execute requires --until-complete for model queues")

    model = profile["experiment"]["model_slug"]
    state_dir = args.state_dir or Path("logs/runs") / args.run_id / model
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "queue_state.json"
    events_path = state_dir / "queue.events.jsonl"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"completed": {}}
    )
    raw_completed = state.get("completed", {})
    completed: dict[str, str] = (
        {str(key): str(value) for key, value in raw_completed.items()}
        if isinstance(raw_completed, dict)
        else {}
    )

    for job in jobs:
        job_id = str(job["id"])
        job_signature = _job_signature(job, args.model_config)
        event_context = {
            "run_id": args.run_id,
            "model": model,
            "dataset": job.get("dataset", "unknown"),
            "stage": job.get("stage", "unknown"),
            "job_id": job_id,
            "job_signature": job_signature,
        }
        if _output_evidence(job) is not None and completed.get(job_id) == job_signature:
            _event(events_path, event="job_reused", **event_context)
            continue
        if job.get("wait_for"):
            dependency = Path(job["wait_for"])
            _event(
                events_path,
                event="dependency_wait",
                path=str(dependency),
                **event_context,
            )
            _wait_for(
                dependency,
                timeout=args.dependency_timeout_seconds,
                poll=args.dependency_poll_seconds,
                expected_run_id=args.run_id,
            )
        _event(events_path, event="job_started", **event_context)
        subprocess.run(job["command"], check=True)
        completed[job_id] = job_signature
        state = {
            "run_id": args.run_id,
            "model": model,
            "completed": dict(sorted(completed.items())),
            "last_job": job_id,
        }
        atomic_write_json(state_path, state)
        _event(events_path, event="job_completed", **event_context)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Wait for DF2023/COFINFAD, then resume the durable Data Fusion 2022 queue."""
from __future__ import annotations

import argparse
import atexit
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cv_llm_queue import valid_dataset_completion
from src.experiments.artifacts import atomic_write_json, fingerprint
from src.utils.process_lease import ProcessLease


ISOLATED_DATASETS = [
    "datafusion_default_2023",
    "cofinfad_operational_fidelity",
]


def isolated_completion_path(run_id: str, model: str) -> Path:
    return (
        Path("logs/runs") / run_id / "completion"
        / f"{model}_isolated_night.json"
    )


def valid_isolated_completion(path: Path, *, run_id: str, model: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    signature = payload.get("completion_signature")
    unsigned = {
        key: value for key, value in payload.items()
        if key != "completion_signature"
    }
    return (
        payload.get("status") == "completed"
        and payload.get("run_id") == run_id
        and payload.get("model") == model
        and payload.get("datasets") == ISOLATED_DATASETS
        and bool(signature)
        and fingerprint(unsigned) == signature
    )


def resume_command(
    *, python_bin: Path, model: str, resume_run_id: str
) -> list[str]:
    return [
        str(python_bin),
        "scripts/run_cv_queue_watchdog.py",
        "--python-bin", str(python_bin),
        "--model", model,
        "--run-id", resume_run_id,
        "--datasets", "datafusion_education",
    ]


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"time": time.time(), **payload}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "gpt_oss"), required=True)
    parser.add_argument(
        "--isolated-run-id", default="reviewer-v11-isolated-r2"
    )
    parser.add_argument(
        "--resume-run-id", default="reviewer-v10-datafusion-public"
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    barriers = {
        model: isolated_completion_path(args.isolated_run_id, model)
        for model in ("qwen", "gpt_oss")
    }
    command = resume_command(
        python_bin=args.python_bin,
        model=args.model,
        resume_run_id=args.resume_run_id,
    )
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "model": args.model,
        "ordering": [
            "datafusion_default_2023",
            "cofinfad_operational_fidelity",
            "datafusion_education_2022_resume",
        ],
        "barrier": "both source models complete DF2023 and COFINFAD",
        "barrier_paths": {key: str(value) for key, value in barriers.items()},
        "resume_run_id": args.resume_run_id,
        "command": command,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return

    event_path = (
        Path("logs/runs") / args.isolated_run_id
        / f"{args.model}.post_isolated_df2022.events.jsonl"
    )
    lease = ProcessLease(
        Path("logs/api_queue_leases")
        / f"post_isolated_df2022_{args.model}.lock",
        {
            "kind": "post_isolated_datafusion2022",
            "model": args.model,
            "isolated_run_id": args.isolated_run_id,
            "resume_run_id": args.resume_run_id,
        },
    )
    lease.acquire()
    atexit.register(lease.release)
    while not all(
        valid_isolated_completion(path, run_id=args.isolated_run_id, model=model)
        for model, path in barriers.items()
    ):
        append_event(event_path, {
            "state": "waiting_for_both_cofinfad_completions",
            "ready": {
                model: valid_isolated_completion(
                    path, run_id=args.isolated_run_id, model=model
                )
                for model, path in barriers.items()
            },
        })
        time.sleep(max(args.poll_seconds, 1.0))

    append_event(event_path, {
        "state": "datafusion2022_resume_started",
        "resume_run_id": args.resume_run_id,
        "command_signature": fingerprint(command),
    })
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    completion = (
        Path("logs/runs") / args.resume_run_id / "completion"
        / f"{args.model}_datafusion_education_dataset.json"
    )
    if not valid_dataset_completion(
        completion,
        run_id=args.resume_run_id,
        dataset="datafusion_education",
        model=args.model,
    ):
        raise RuntimeError(
            "Data Fusion 2022 queue exited without a valid dataset completion"
        )
    payload = {
        "status": "completed",
        "model": args.model,
        "isolated_run_id": args.isolated_run_id,
        "resume_run_id": args.resume_run_id,
        "datafusion2022_completion": str(completion),
    }
    payload["completion_signature"] = fingerprint(payload)
    marker = (
        Path("logs/runs") / args.isolated_run_id / "completion"
        / f"{args.model}_post_isolated_datafusion2022.json"
    )
    atomic_write_json(marker, payload)
    append_event(event_path, {"state": "completed", "marker": str(marker)})


if __name__ == "__main__":
    main()

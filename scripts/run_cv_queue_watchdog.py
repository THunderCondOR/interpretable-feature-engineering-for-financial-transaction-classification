#!/usr/bin/env python3
"""Restart a CV API queue after a fully timed-out atomic window."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

RESTARTABLE_CONNECTION_ERRORS = {
    "APITimeoutError",
    "TimeoutError",
    "APIConnectionError",
    "ConnectionError",
}


class WatchdogShutdown(Exception):
    """Internal control flow used to clean up the child process on signals."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"watchdog received signal {signum}")
        self.signum = int(signum)


def request_shutdown(signum: int, _frame: Any) -> None:
    """Turn a terminal signal into a cleanup-aware exception."""
    raise WatchdogShutdown(signum)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.time(), **payload}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def full_connection_failure(
    event: dict[str, Any],
    window_sizes: dict[str, int],
) -> bool:
    """Return true only when every request in an atomic window lost connection."""
    if (
        event.get("event") != "window_rolled_back"
        or event.get("reason") != "transient_or_incomplete"
    ):
        return False
    errors = event.get("errors")
    if not isinstance(errors, dict) or not errors:
        return False
    if not set(errors).issubset(RESTARTABLE_CONNECTION_ERRORS):
        return False
    batch_id = str(event.get("batch_id") or "")
    window_size = window_sizes.get(batch_id)
    if not window_size:
        return False
    return sum(int(count) for count in errors.values()) >= window_size


def consume_events(
    path: Path,
    offset: int,
    window_sizes: dict[str, int],
) -> tuple[int, bool, dict[str, Any] | None]:
    """Read appended scheduler events and detect a poisoned connection window."""
    if not path.is_file():
        return offset, False, None
    restart_event = None
    with path.open("r", encoding="utf-8") as file:
        file.seek(offset)
        for line in file:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            batch_id = str(event.get("batch_id") or "")
            if event.get("event") == "window_started" and batch_id:
                window_sizes[batch_id] = int(event.get("size") or 0)
            if full_connection_failure(event, window_sizes):
                restart_event = event
        offset = file.tell()
    return offset, restart_event is not None, restart_event


def terminate_process_group(
    process: subprocess.Popen,
    *,
    grace_seconds: float,
) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def queue_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python_bin),
        "scripts/run_cv_llm_queue.py",
        "--model",
        args.model,
        "--run-id",
        args.run_id,
        "--datasets",
        args.datasets,
        "--execute",
        "--execute-api",
        "--until-complete",
        "--fast-resume",
    ]
    if args.run_pilots:
        command.append("--run-pilots")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "gpt_oss"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--datasets",
        default="berka,datafusion_education",
    )
    parser.add_argument("--run-pilots", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--restart-cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=20.0)
    args = parser.parse_args()

    scheduler_events = (
        REPO_ROOT / "logs" / "runs" / args.run_id
        / f"{args.model}.events.jsonl"
    )
    watchdog_events = (
        REPO_ROOT / "logs" / "runs" / args.run_id
        / f"{args.model}.watchdog.events.jsonl"
    )
    command = queue_command(args)
    restart_count = 0
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_shutdown)

    process: subprocess.Popen | None = None
    try:
        while True:
            offset = scheduler_events.stat().st_size if scheduler_events.is_file() else 0
            window_sizes: dict[str, int] = {}
            append_event(watchdog_events, {
                "event": "queue_process_started",
                "model": args.model,
                "restart_count": restart_count,
                "command": command,
            })
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                start_new_session=True,
            )
            restart_reason = None
            while process.poll() is None:
                time.sleep(max(args.poll_seconds, 0.1))
                offset, should_restart, event = consume_events(
                    scheduler_events,
                    offset,
                    window_sizes,
                )
                if should_restart:
                    restart_reason = event
                    append_event(watchdog_events, {
                        "event": "full_connection_window_detected",
                        "model": args.model,
                        "restart_count": restart_count,
                        "batch_id": event.get("batch_id") if event else None,
                        "errors": event.get("errors") if event else None,
                    })
                    terminate_process_group(
                        process,
                        grace_seconds=args.terminate_grace_seconds,
                    )
                    break

            returncode = process.wait()
            process = None
            if restart_reason is None:
                append_event(watchdog_events, {
                    "event": "queue_process_exited",
                    "model": args.model,
                    "restart_count": restart_count,
                    "returncode": returncode,
                })
                raise SystemExit(returncode)

            restart_count += 1
            append_event(watchdog_events, {
                "event": "queue_restart_cooldown",
                "model": args.model,
                "restart_count": restart_count,
                "seconds": args.restart_cooldown_seconds,
            })
            time.sleep(max(args.restart_cooldown_seconds, 0.0))
    except WatchdogShutdown as exc:
        # Ignore additional terminal signals while the child group receives
        # TERM and, if necessary, KILL after its grace period.
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        if process is not None and process.poll() is None:
            terminate_process_group(
                process,
                grace_seconds=args.terminate_grace_seconds,
            )
        append_event(watchdog_events, {
            "event": "watchdog_shutdown",
            "model": args.model,
            "restart_count": restart_count,
            "signal": exc.signum,
            "child_stopped": process is None or process.poll() is not None,
        })
        raise SystemExit(128 + exc.signum) from None


if __name__ == "__main__":
    main()

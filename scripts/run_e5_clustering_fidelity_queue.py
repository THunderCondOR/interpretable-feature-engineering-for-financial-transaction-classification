#!/usr/bin/env python3
"""Run clustering selection, E5 rebuild, and fidelity as one resumable queue."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json


STAGES = (
    (
        "clustering_selection",
        [
            "scripts/run_rosbank_clustering_sweep.py",
            "--skip-reference", "--execute",
        ],
        Path("results/v2/derived/rosbank-clustering-sweep-e5-v1/selection.json"),
    ),
    (
        "e5_rebuild",
        ["scripts/run_e5_rebuild_suite.py", "--execute"],
        Path(
            "results/v2/derived/reviewer-v6-e5-clustering/rebuild_state.json"
        ),
    ),
    (
        "fidelity",
        ["scripts/run_fidelity_suite.py", "--execute"],
        Path("results/v2/derived/fidelity-v2"),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("results/v2/derived/reviewer-v6-queue.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "stages": [
            {"name": name, "command": command, "expected": str(expected)}
            for name, command, expected in STAGES
        ],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return

    state = {
        **plan,
        "state": "running",
        "completed": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.state.is_file():
        previous = json.loads(args.state.read_text(encoding="utf-8"))
        state["completed"] = previous.get("completed", [])
    completed = set(state["completed"])
    atomic_write_json(args.state, state)
    for name, command, expected in STAGES:
        if name in completed and expected.exists():
            continue
        state["current"] = name
        atomic_write_json(args.state, state)
        subprocess.run(
            [sys.executable, *command], cwd=REPO_ROOT, check=True
        )
        if not expected.exists():
            raise FileNotFoundError(
                f"Stage {name} did not create expected artifact: {expected}"
            )
        state["completed"].append(name)
        completed.add(name)
        atomic_write_json(args.state, state)
    state["state"] = "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(args.state, state)


if __name__ == "__main__":
    main()

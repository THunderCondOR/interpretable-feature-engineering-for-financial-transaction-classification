#!/usr/bin/env python3
"""Run validation-only cluster tuning, full rebuild, and fidelity-v2."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json

STAGES = (
    (
        "cluster_quality_sweep",
        ["scripts/run_cluster_quality_sweep.py", "--dataset", "rosbank", "--execute"],
        Path("results/v2/derived/cluster-quality-v2/rosbank/selection.json"),
    ),
    (
        "cluster_quality_rebuild",
        ["scripts/run_cluster_quality_rebuild.py", "--execute"],
        Path(
            "results/v2/derived/reviewer-v7-cluster-quality/rebuild_state.json"
        ),
    ),
    (
        "fidelity_v2",
        [
            "scripts/run_fidelity_suite.py",
            "--rebuild-root",
            "results/v2/derived/reviewer-v7-cluster-quality",
            "--output-root",
            "results/v2/derived/fidelity-v2",
            "--execute",
        ],
        Path("results/v2/derived/fidelity-v2/fidelity_state.json"),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=Path,
        default=Path(
            "results/v2/derived/cluster-quality-fidelity-v2-queue.json"
        ),
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
    state = {**plan, "state": "running", "completed": []}
    if args.state.is_file():
        state["completed"] = json.loads(
            args.state.read_text(encoding="utf-8")
        ).get("completed", [])
    completed = set(state["completed"])
    for name, command, expected in STAGES:
        if name in completed and expected.is_file():
            continue
        state["current"] = name
        atomic_write_json(args.state, state)
        try:
            subprocess.run(
                [sys.executable, *command], cwd=REPO_ROOT, check=True
            )
            if not expected.is_file():
                raise FileNotFoundError(expected)
        except Exception as error:
            state["state"] = "failed"
            state["error"] = {
                "stage": name,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(args.state, state)
            raise
        completed.add(name)
        state["completed"] = sorted(completed)
        atomic_write_json(args.state, state)
    state["state"] = "completed"
    state["current"] = None
    state.pop("error", None)
    atomic_write_json(args.state, state)


if __name__ == "__main__":
    main()

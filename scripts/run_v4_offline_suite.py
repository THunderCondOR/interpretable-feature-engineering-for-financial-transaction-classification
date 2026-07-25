#!/usr/bin/env python3
"""Run the v4 clustering/feature queue sequentially across immutable API cells."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(
    "/home/chaichuk/miniconda3/envs/"
    "breaking-the-chain-env/bin/python"
)

CELLS = (
    (
        "rosbank",
        "gpt_oss",
        Path("results/v2/rosbank/guided_zero_shot_v4/gpt_oss/seed_17"),
    ),
    (
        "gender",
        "gpt_oss",
        Path("results/v2/gender/guided_zero_shot_v4/gpt_oss/seed_17"),
    ),
    (
        "gender",
        "qwen",
        Path("results/v2/gender/guided_zero_shot_v4/qwen/seed_17"),
    ),
    (
        "age",
        "gpt_oss",
        Path(
            "results/v2/age/"
            "guided_zero_shot_v4__age_opaque/gpt_oss/seed_17"
        ),
    ),
    (
        "age",
        "qwen",
        Path(
            "results/v2/age/"
            "guided_zero_shot_v4__age_opaque/qwen/seed_17"
        ),
    ),
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def cell_key(dataset: str, model: str) -> str:
    return f"{dataset}:{model}"


def build_jobs(skip: set[str]) -> list[dict[str, str]]:
    jobs = []
    for dataset, model, source in CELLS:
        key = cell_key(dataset, model)
        if key in skip:
            continue
        if not (source / "manifest.json").is_file():
            raise FileNotFoundError(f"Missing immutable source cell: {source}")
        jobs.append({
            "key": key,
            "dataset": dataset,
            "model": model,
            "source_root": str(source),
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("results/v2/derived/reviewer-v4-offline-v2"),
    )
    parser.add_argument(
        "--skip-cell",
        action="append",
        default=["rosbank:qwen"],
        help="Dataset:model cell already handled elsewhere; repeatable.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(set(args.skip_cell))
    status_path = args.derived_root / "queue_status.json"
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "policy": "sequential; stop on first failure; child stages resume safely",
        "ml": False,
        "jobs": jobs,
        "status_path": str(status_path),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    state = {
        **plan,
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed": [],
        "failed": None,
    }
    atomic_json(status_path, state)
    for job in jobs:
        state["current"] = job["key"]
        atomic_json(status_path, state)
        command = [
            str(PYTHON),
            "scripts/run_v4_offline_pipeline.py",
            "--source-root",
            job["source_root"],
            "--derived-root",
            str(args.derived_root),
            "--clustering-backend",
            "auto",
            "--skip-ml",
            "--execute",
        ]
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            state["state"] = "failed"
            state["failed"] = {
                "cell": job["key"],
                "returncode": error.returncode,
            }
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(status_path, state)
            raise
        state["completed"].append(job["key"])
        atomic_json(status_path, state)

    state["state"] = "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(status_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

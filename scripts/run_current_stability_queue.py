#!/usr/bin/env python3
"""Resumable stability queue for final E5 artifacts across all datasets."""
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


V2_ROOT = Path("results/v2/derived/reviewer-v6-e5-clustering")
ROSBANK_E5_CACHE = Path(
    "results/v2/derived/rosbank-embedding-sweep-v1/"
    "multilingual_e5_large/rosbank"
)
BERKA_ROOT = Path("results/v5/derived/cv_main_e5")


def run(command: list[str]) -> None:
    subprocess.run([sys.executable, *command], cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datafusion-run-id", default="reviewer-v10-datafusion-public")
    parser.add_argument("--wait-for-tmux-session", default="reviewer_v10_lora_all_datasets")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--state", type=Path,
        default=Path("results/v5/derived/reviewer-v10-stability-queue.json"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    v2_cells = [
        (dataset, model)
        for dataset in ("rosbank", "gender", "age")
        for model in ("qwen", "gpt_oss")
    ]
    stages = ["wait_lora_gpu"]
    for dataset, model in v2_cells:
        stages.extend([
            f"v2_cluster_seeds_{dataset}_{model}",
            f"v2_granularity_{dataset}_{model}",
        ])
    stages.extend([
        "berka_cluster_stability",
        "resume_lora",
    ])
    plan = {
        "mode": "execute" if args.execute else "dry-run", "stages": stages,
        "seeds": [17, 101, 947], "cluster_counts": [100, 200, 400, 800],
        "embedding_model": "intfloat/multilingual-e5-large",
        "datafusion_protocol": "public_kfold5_seed100",
        "datafusion_run_id": args.datafusion_run_id,
        "priority": "current stability, then LoRA without waiting for DataFusion API",
        "resume_after_stability": "reviewer_v10_lora_all_datasets",
        "datafusion_priority_waiter": "reviewer_v10_datafusion_gpu_waiter",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    previous = json.loads(args.state.read_text()) if args.state.is_file() else {}
    completed = list(previous.get("completed", []))
    state = {**plan, "state": "running", "completed": completed,
             "started_at": previous.get("started_at", datetime.now(timezone.utc).isoformat())}
    atomic_write_json(args.state, state)

    def stage(name: str, callback) -> None:
        if name in completed:
            return
        state["current"] = name
        atomic_write_json(args.state, state)
        callback()
        completed.append(name)
        state["completed"] = completed
        atomic_write_json(args.state, state)

    def wait_lora() -> None:
        while subprocess.run(
            ["tmux", "has-session", "-t", args.wait_for_tmux_session],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0:
            time.sleep(max(5.0, args.poll_seconds))

    stage("wait_lora_gpu", wait_lora)

    for dataset, model in v2_cells:
        def cluster_seed_command(dataset=dataset, model=model) -> list[str]:
            command = [
            "scripts/run_v4_cluster_seed_stability.py",
            "--derived-root", str(V2_ROOT),
            "--datasets", dataset,
            "--model", model,
            "--seeds", "17", "101", "947",
            "--execute",
            ]
            if dataset == "rosbank":
                command[1:1] = [
                    "--embedding-cache-cell",
                    str(ROSBANK_E5_CACHE / model / "seed_17"),
                ]
            return command

        stage(
            f"v2_cluster_seeds_{dataset}_{model}",
            lambda command=cluster_seed_command(): run(command),
        )
        stage(f"v2_granularity_{dataset}_{model}", lambda dataset=dataset, model=model: run([
            "scripts/run_v4_stability.py",
            "--derived-cell", str(V2_ROOT / dataset / model / "seed_17"),
            "--execute",
        ]))

    stage("berka_cluster_stability", lambda: run([
        "scripts/run_cv_cluster_stability.py", "--dataset", "berka",
        "--run-id", "reviewer-v5-fixed-new-datasets",
        "--derived-root", str(BERKA_ROOT), "--execute",
    ]))

    stage("resume_lora", lambda: subprocess.run(
        [
            "bash", "scripts/launch_all_lora_queue.sh",
            "--run-id", "reviewer-v10-lora-all-datasets", "--execute",
        ],
        cwd=REPO_ROOT,
        check=True,
    ))
    state["state"] = "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(args.state, state)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resumable stability queue for final E5 artifacts across all datasets."""
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


V2_ROOT = Path("results/v2/derived/reviewer-v7-cluster-quality")
BERKA_ROOT = Path("results/v5/derived/cv_main_e5")
DATAFUSION_ROOT = Path("results/v5/derived/cv_main_e5_public")


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
    stages = [
        "wait_lora_gpu", "v2_cluster_seeds", "v2_granularity",
        "berka_cluster_stability", "wait_datafusion_api",
        "datafusion_offline", "datafusion_cluster_stability",
    ]
    plan = {
        "mode": "execute" if args.execute else "dry-run", "stages": stages,
        "seeds": [17, 101, 947], "cluster_counts": [100, 200, 400, 800],
        "embedding_model": "intfloat/multilingual-e5-large",
        "datafusion_protocol": "public_kfold5_seed100",
        "datafusion_run_id": args.datafusion_run_id,
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
    stage("v2_cluster_seeds", lambda: run([
        "scripts/run_v4_cluster_seed_stability.py", "--derived-root", str(V2_ROOT),
        "--seeds", "17", "101", "947", "--execute",
    ]))

    def v2_granularity() -> None:
        for dataset in ("rosbank", "gender", "age"):
            for model in ("qwen", "gpt_oss"):
                run([
                    "scripts/run_v4_stability.py", "--derived-cell",
                    str(V2_ROOT / dataset / model / "seed_17"), "--execute",
                ])
    stage("v2_granularity", v2_granularity)
    stage("berka_cluster_stability", lambda: run([
        "scripts/run_cv_cluster_stability.py", "--dataset", "berka",
        "--run-id", "reviewer-v5-fixed-new-datasets",
        "--derived-root", str(BERKA_ROOT), "--execute",
    ]))

    def wait_datafusion() -> None:
        while True:
            ready = all(valid_dataset_completion(
                Path("logs/runs") / args.datafusion_run_id / "completion"
                / f"{model}_datafusion_education_dataset.json",
                run_id=args.datafusion_run_id, dataset="datafusion_education",
                model=model,
            ) for model in ("qwen", "gpt_oss"))
            if ready:
                return
            time.sleep(max(5.0, args.poll_seconds))
    stage("wait_datafusion_api", wait_datafusion)
    stage("datafusion_offline", lambda: run([
        "scripts/run_cv_offline_pipeline.py", "--datasets", "datafusion_education",
        "--models", "qwen,gpt_oss", "--folds", "0,1,2,3,4",
        "--run-id", args.datafusion_run_id, "--derived-root", str(DATAFUSION_ROOT),
        "--embedding-model", "intfloat/multilingual-e5-large", "--execute",
    ]))
    stage("datafusion_cluster_stability", lambda: run([
        "scripts/run_cv_cluster_stability.py", "--dataset", "datafusion_education",
        "--run-id", args.datafusion_run_id, "--derived-root", str(DATAFUSION_ROOT),
        "--execute",
    ]))
    state["state"] = "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(args.state, state)


if __name__ == "__main__":
    main()

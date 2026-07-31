#!/usr/bin/env python3
"""Wait for one model's Data Fusion queue, then resume full LLM seed runs."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_cv_llm_queue import valid_dataset_completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "gpt_oss"), required=True)
    parser.add_argument("--datafusion-run-id", default="reviewer-v10-datafusion-public")
    parser.add_argument("--stability-run-id", default="reviewer-v4-llm-full-seeds-v2")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    model_config = Path("configs/v2") / f"{args.model}.yaml"
    results_root = Path("results/v2/stability/llm-full-seeds-v2")
    print({
        "mode": "execute" if args.execute else "dry-run", "model": args.model,
        "wait_for": f"{args.datafusion_run_id}/{args.model}/datafusion",
        "generation_seeds": [101, 947], "scope": "full",
        "results_root": str(results_root),
    }, flush=True)
    if not args.execute:
        return
    marker = (
        Path("logs/runs") / args.datafusion_run_id / "completion"
        / f"{args.model}_datafusion_education_dataset.json"
    )
    while not valid_dataset_completion(
        marker, run_id=args.datafusion_run_id,
        dataset="datafusion_education", model=args.model,
    ):
        time.sleep(max(5.0, args.poll_seconds))
    subprocess.run([
        sys.executable, "scripts/run_llm_seed_stability.py",
        "--model-config", str(model_config), "--run-id", args.stability_run_id,
        "--scope", "full", "--seeds", "101", "947",
        "--results-root", str(results_root),
        "--execute", "--execute-api", "--until-complete",
    ], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

"""Sequential per-model queue; dry-run unless both execution guards are set."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import yaml


def default_jobs(profile):
    model = profile["experiment"]["model_slug"]
    common = {"model": model, "run_id": profile["experiment"]["run_id"]}
    if model == "qwen":
        jobs = [
            {**common, "dataset": "gender", "stage": "pilot", "variant": variant, "api": variant != "legacy"}
            for variant in ["legacy", "neutral_only", "neutral_robust_fewshot", "neutral_robust_zero_shot"]
        ]
        jobs += [{**common, "dataset": dataset, "stage": stage, "api": False}
            for dataset in ["gender", "age", "rosbank"] for stage in ["robust_stats", "offline_stability", "seeded_ml"]]
    else:
        jobs = [
            {**common, "dataset": "gender", "stage": "selected_seed_subset", "api": True, "blocked_by": "qwen_prompt_selection"},
            {**common, "dataset": "gender", "stage": "full_test_direct", "api": True, "blocked_by": "qwen_prompt_selection"},
            {**common, "dataset": "gender", "stage": "grounding_claims", "api": True, "blocked_by": "qwen_prompt_selection"},
        ]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    profile = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    profile["experiment"]["run_id"] = args.run_id
    jobs = json.loads(args.queue.read_text()) if args.queue else default_jobs(profile)
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", "profile": str(args.model_config), "jobs": jobs}, indent=2))
    if not args.execute:
        return
    if not args.until_complete:
        raise ValueError("--execute requires --until-complete for model queues")
    command_jobs = [job for job in jobs if job.get("command")]
    unresolved = [job for job in jobs if not job.get("command") and not job.get("blocked_by")]
    if unresolved:
        raise RuntimeError("Queue contains planned cells without resolved commands; materialize gender selection and paths first")
    for job in command_jobs:
        subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()

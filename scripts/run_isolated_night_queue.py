#!/usr/bin/env python3
"""Run DF2023 then COFINFAD sequentially for one model."""

from __future__ import annotations

import argparse
import atexit
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.common import load_json
from src.experiments.artifacts import atomic_write_json, fingerprint
from src.experiments.config_builder import load_yaml
from src.utils.process_lease import ProcessLease


BENCHMARKS = (
    ("datafusion_default_2023", Path("configs/datafusion_default_2023/base.yaml")),
    ("cofinfad_operational_fidelity", Path("configs/cofinfad/base.yaml")),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "gpt_oss"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/isolated_benchmarks"))
    parser.add_argument("--results-root", type=Path, default=Path("results/isolated"))
    parser.add_argument("--generated-root", type=Path, default=Path("logs/runs/isolated/generated"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    jobs = []
    for dataset, config_path in BENCHMARKS:
        base = load_yaml(config_path)
        protocol = base["dataset"]["benchmark_protocol"]
        manifest = args.data_root / dataset / protocol / "benchmark_manifest.json"
        command = [
            sys.executable, "scripts/run_isolated_model_queue.py", "--model", args.model,
            "--manifest", str(manifest), "--base-config", str(config_path),
            "--run-id", args.run_id, "--results-root", str(args.results_root),
            "--generated-root", str(args.generated_root),
        ]
        if args.execute:
            command.extend(["--execute", "--execute-api", "--until-complete"])
        jobs.append({"dataset": dataset, "manifest": str(manifest), "command": command})
    plan = {
        "mode": "execute" if args.execute else "dry-run", "run_id": args.run_id,
        "model": args.model, "ordering": [job["dataset"] for job in jobs], "jobs": jobs,
        "model_parallelism": "independent qwen and gpt_oss launchers",
        "dataset_parallelism_within_model": "strictly sequential",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if not args.execute_api or not args.until_complete:
        raise ValueError("Execution requires every API safety guard")
    for job in jobs:
        manifest = Path(job["manifest"])
        if not manifest.is_file():
            raise FileNotFoundError(f"Prepare benchmark first: {manifest}")
        if load_json(manifest).get("dataset") != job["dataset"]:
            raise ValueError(f"Manifest mismatch: {manifest}")
    lease = ProcessLease(
        Path("logs/api_queue_leases") / f"isolated_night_{args.model}.lock",
        {"kind": "isolated_night_queue", "model": args.model, "run_id": args.run_id},
    )
    lease.acquire()
    atexit.register(lease.release)
    completed = []
    for job in jobs:
        subprocess.run(job["command"], cwd=REPO_ROOT, check=True)
        completed.append(job["dataset"])
    payload = {"status": "completed", "run_id": args.run_id, "model": args.model, "datasets": completed}
    payload["completion_signature"] = fingerprint(payload)
    atomic_write_json(Path("logs/runs") / args.run_id / "completion" / f"{args.model}_isolated_night.json", payload)


if __name__ == "__main__":
    main()

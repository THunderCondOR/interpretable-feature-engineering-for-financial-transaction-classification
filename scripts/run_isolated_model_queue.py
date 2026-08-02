#!/usr/bin/env python3
"""Run one model's durable queue for one isolated benchmark."""

from __future__ import annotations

import argparse
import atexit
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.runtime import load_benchmark_manifest
from src.experiments.artifacts import atomic_write_json, file_sha256, fingerprint
from src.experiments.config_builder import load_yaml, slug
from src.utils.process_lease import ProcessLease


def append_event(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.time(), **payload}, ensure_ascii=False) + "\n")


def valid_selection(path: Path, *, run_id: str, dataset: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    claimed = payload.get("selection_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "selection_sha256"}
    if (
        payload.get("status") != "completed"
        or payload.get("run_id") != run_id
        or payload.get("dataset") != dataset
        or not claimed
        or fingerprint(unsigned) != claimed
    ):
        return False
    for entry in payload.get("selected_configs", {}).values():
        config = Path(str(entry.get("path", "")))
        if not config.is_file() or entry.get("sha256") != file_sha256(config):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "gpt_oss"), required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/isolated"))
    parser.add_argument("--generated-root", type=Path, default=Path("logs/runs/isolated/generated"))
    parser.add_argument("--qwen-config", type=Path, default=Path("configs/v2/qwen.yaml"))
    parser.add_argument("--gpt-config", type=Path, default=Path("configs/v2/gpt_oss.yaml"))
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    base = load_yaml(args.base_config)
    dataset = base["dataset"]["name"]
    manifest = load_benchmark_manifest(args.manifest, dataset)
    cell = args.generated_root / slug(args.run_id) / dataset
    selection = cell / "prompt_selection.json"
    profile = args.qwen_config if args.model == "qwen" else args.gpt_config
    selected_config = cell / f"selected_{args.model}.yaml"
    completion = Path("logs/runs") / args.run_id / "completion" / f"{args.model}_{dataset}.json"
    pilot_command = [
        sys.executable, "scripts/run_isolated_prompt_pilot.py",
        "--manifest", str(args.manifest), "--base-config", str(args.base_config),
        "--run-id", args.run_id, "--results-root", str(args.results_root),
        "--generated-root", str(args.generated_root),
        "--qwen-config", str(args.qwen_config), "--gpt-config", str(args.gpt_config),
        "--stage", "all", "--execute", "--execute-api", "--until-complete",
    ]
    full_command = [
        sys.executable, "scripts/run_full_llm_generation.py",
        "--dataset", dataset, "--base-config", str(args.base_config),
        "--model-config", str(profile), "--selected-config", str(selected_config),
        "--selection", str(selection), "--splits", "train,val,test",
        "--run-id", args.run_id, "--results-root", str(args.results_root),
        "--completion-marker", str(completion), "--fast-resume",
        "--execute", "--execute-api", "--until-complete",
    ]
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "dataset": dataset,
        "model": args.model,
        "run_id": args.run_id,
        "pilot_owner": "qwen",
        "expected_clients": {role: manifest["counts"][role] for role in ("train", "val", "test")},
        "expected_main_requests": 2 * sum(manifest["counts"][role] for role in ("train", "val", "test")),
        "pilot_command": pilot_command if args.model == "qwen" else None,
        "full_command": full_command,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if not args.execute_api or not args.until_complete:
        raise ValueError("Execution requires --execute-api and --until-complete")
    lease = ProcessLease(
        Path("logs/api_queue_leases") / f"isolated_{dataset}_{args.model}.lock",
        {"kind": "isolated_api_queue", "dataset": dataset, "model": args.model, "run_id": args.run_id},
    )
    lease.acquire()
    atexit.register(lease.release)
    events = Path("logs/runs") / args.run_id / f"{args.model}.isolated_queue.events.jsonl"
    if args.model == "qwen" and not valid_selection(selection, run_id=args.run_id, dataset=dataset):
        append_event(events, state="pilot_started", dataset=dataset)
        subprocess.run(pilot_command, cwd=REPO_ROOT, check=True)
        append_event(events, state="pilot_completed", dataset=dataset)
    while not valid_selection(selection, run_id=args.run_id, dataset=dataset):
        append_event(events, state="waiting_for_prompt_selection", dataset=dataset)
        time.sleep(max(args.wait_seconds, 1.0))
    append_event(events, state="full_started", dataset=dataset, command_signature=fingerprint(full_command))
    subprocess.run(full_command, cwd=REPO_ROOT, check=True)
    if not completion.is_file():
        raise RuntimeError("Full runner exited without completion marker")
    append_event(events, state="completed", dataset=dataset, completion=str(completion))
    atomic_write_json(
        Path("logs/runs") / args.run_id / "completion" / f"{args.model}_{dataset}_queue.json",
        {"status": "completed", "run_id": args.run_id, "dataset": dataset, "model": args.model, "completion": str(completion)},
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Durable per-model API queue for the two fold-based v5 benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.benchmark_registry import BENCHMARKS
from src.experiments.artifacts import atomic_write_json, fingerprint


DATASET_ORDER = ("berka", "datafusion_education")
FOLDS = range(5)


def selection_path(run_id: str, dataset: str, fold: int) -> Path:
    return (
        Path("logs/runs") / run_id / "generated" / dataset / f"fold_{fold}"
        / "prompt_selection.json"
    )


def selected_config_path(
    run_id: str, dataset: str, fold: int, model: str
) -> Path:
    return (
        Path("logs/runs") / run_id / "generated" / dataset / f"fold_{fold}"
        / f"selected_{model}.yaml"
    )


def completion_path(
    run_id: str, dataset: str, fold: int, model: str
) -> Path:
    return (
        Path("logs/runs") / run_id / "completion"
        / f"{model}_{dataset}_fold_{fold}.json"
    )


def pilot_command(
    *,
    dataset: str,
    fold: int,
    run_id: str,
    qwen_config: Path,
    gpt_config: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_cv_prompt_pilots.py",
        "--dataset", dataset,
        "--folds", str(fold),
        "--run-id", run_id,
        "--qwen-config", str(qwen_config),
        "--gpt-config", str(gpt_config),
        "--stage", "all",
        "--execute", "--execute-api", "--until-complete",
    ]


def full_command(
    *,
    dataset: str,
    fold: int,
    model: str,
    run_id: str,
    model_config: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_full_llm_generation.py",
        "--dataset", dataset,
        "--base-config", f"configs/v5/{dataset}.yaml",
        "--model-config", str(model_config),
        "--selected-config",
        str(selected_config_path(run_id, dataset, fold, model)),
        "--selection", str(selection_path(run_id, dataset, fold)),
        "--splits", "train,test",
        "--run-id", run_id,
        "--completion-marker",
        str(completion_path(run_id, dataset, fold, model)),
        "--execute", "--execute-api", "--until-complete",
    ]


def jobs(
    *,
    model: str,
    run_id: str,
    qwen_config: Path,
    gpt_config: Path,
    datasets: tuple[str, ...] = DATASET_ORDER,
    run_pilots: bool = False,
) -> list[dict[str, Any]]:
    result = []
    profile = qwen_config if model == "qwen" else gpt_config
    for dataset in datasets:
        for fold in FOLDS:
            selection = selection_path(run_id, dataset, fold)
            if model == "qwen" or run_pilots:
                result.append({
                    "id": f"{dataset}_fold_{fold}_pilot",
                    "kind": "pilot",
                    "dataset": dataset,
                    "fold": fold,
                    "expected_output": str(selection),
                    "command": pilot_command(
                        dataset=dataset,
                        fold=fold,
                        run_id=run_id,
                        # run_cv_prompt_pilots historically calls the selector
                        # profile "qwen".  When Qwen is unavailable, explicitly
                        # pass the active GPT profile as selector instead.
                        qwen_config=profile if run_pilots else qwen_config,
                        gpt_config=gpt_config,
                    ),
                })
            result.append({
                "id": f"{dataset}_fold_{fold}_{model}_full",
                "kind": "full",
                "dataset": dataset,
                "fold": fold,
                "wait_for": str(selection),
                "expected_output": str(
                    completion_path(run_id, dataset, fold, model)
                ),
                "command": full_command(
                    dataset=dataset,
                    fold=fold,
                    model=model,
                    run_id=run_id,
                    model_config=profile,
                ),
            })
    return result


def valid_output(path: Path, *, run_id: str, job: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("run_id") != run_id:
        return False
    if payload.get("dataset") != job["dataset"]:
        return False
    if payload.get("fold") is None or int(payload["fold"]) != int(job["fold"]):
        return False
    return (
        bool(payload.get("selection_sha256"))
        if job["kind"] == "pilot"
        else payload.get("status") == "completed"
    )


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "gpt_oss"), required=True)
    parser.add_argument("--run-id", default="reviewer-v5-benchmarks")
    parser.add_argument(
        "--qwen-config", type=Path, default=Path("configs/v2/qwen.yaml")
    )
    parser.add_argument(
        "--gpt-config", type=Path, default=Path("configs/v2/gpt_oss.yaml")
    )
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_ORDER),
        help="Comma-separated benchmark queue subset, in execution order.",
    )
    parser.add_argument(
        "--run-pilots",
        action="store_true",
        help=(
            "Run each fold's prompt pilot with this queue's model before the "
            "full cell. Useful when the default Qwen selector is unavailable."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    datasets = tuple(
        value.strip() for value in args.datasets.split(",") if value.strip()
    )
    unknown = set(datasets) - set(DATASET_ORDER)
    if unknown or not datasets:
        raise ValueError(f"Unsupported or empty dataset queue: {sorted(unknown)}")
    queue = jobs(
        model=args.model,
        run_id=args.run_id,
        qwen_config=args.qwen_config,
        gpt_config=args.gpt_config,
        datasets=datasets,
        run_pilots=args.run_pilots,
    )
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "run_id": args.run_id,
        "model": args.model,
        "dataset_order": list(datasets),
        "pilot_selector": args.model if args.run_pilots else "qwen",
        "folds": list(FOLDS),
        "jobs": [
            {
                key: job.get(key)
                for key in ("id", "kind", "dataset", "fold", "wait_for")
            }
            for job in queue
        ],
        "main_api_requests_both_models": 183_820,
        "pilot_requests_qwen": 3 * (5 * 100 + 5 * 400),
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if not args.execute_api or not args.until_complete:
        raise ValueError(
            "Queue execution requires --execute-api --until-complete"
        )
    event_path = (
        Path("logs/runs") / args.run_id / f"{args.model}.cv_queue.events.jsonl"
    )
    for job in queue:
        output = Path(job["expected_output"])
        if valid_output(output, run_id=args.run_id, job=job):
            append_event(event_path, {
                "time": time.time(), "state": "reused", "job": job["id"],
            })
            continue
        dependency = job.get("wait_for")
        while dependency and not Path(dependency).is_file():
            append_event(event_path, {
                "time": time.time(), "state": "waiting",
                "job": job["id"], "dependency": dependency,
            })
            time.sleep(max(args.wait_seconds, 1.0))
        append_event(event_path, {
            "time": time.time(), "state": "started", "job": job["id"],
            "command_signature": fingerprint(job["command"]),
        })
        try:
            subprocess.run(job["command"], cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            append_event(event_path, {
                "time": time.time(), "state": "failed", "job": job["id"],
                "returncode": exc.returncode,
            })
            raise
        if not valid_output(output, run_id=args.run_id, job=job):
            raise RuntimeError(
                f"Job exited without compatible completion: {job['id']}"
            )
        append_event(event_path, {
            "time": time.time(), "state": "completed", "job": job["id"],
        })
    atomic_write_json(
        Path("logs/runs") / args.run_id / "completion"
        / f"{args.model}_cv_queue.json",
        {
            "status": "completed",
            "run_id": args.run_id,
            "model": args.model,
            "jobs": [job["id"] for job in queue],
        },
    )


if __name__ == "__main__":
    main()

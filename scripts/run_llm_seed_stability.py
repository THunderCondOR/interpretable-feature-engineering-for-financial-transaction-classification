#!/usr/bin/env python3
"""Generate fixed-client LLM stability runs for two additional seeds."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.client_sampling import stratified_client_ids
from src.data.loader import add_features, load_dataset
from src.experiments.artifacts import (
    atomic_write_json,
    file_sha256,
    fingerprint,
)
from src.experiments.config_builder import (
    build_runtime_config,
    load_yaml,
    slug,
    write_runtime_config,
)
from scripts.run_full_llm_generation import (
    expected_ids_by_split,
    validate_completion,
)


SOURCE_ROOTS = {
    ("gender", "qwen"): Path(
        "results/v2/gender/guided_zero_shot_v4/qwen/seed_17"
    ),
    ("gender", "gpt_oss"): Path(
        "results/v2/gender/guided_zero_shot_v4/gpt_oss/seed_17"
    ),
    ("rosbank", "qwen"): Path(
        "results/v2/rosbank/guided_zero_shot_v4/qwen/seed_17"
    ),
    ("rosbank", "gpt_oss"): Path(
        "results/v2/rosbank/guided_zero_shot_v4/gpt_oss/seed_17"
    ),
    ("age", "qwen"): Path(
        "results/v2/age/guided_zero_shot_v4__age_opaque/qwen/seed_17"
    ),
    ("age", "gpt_oss"): Path(
        "results/v2/age/guided_zero_shot_v4__age_opaque/gpt_oss/seed_17"
    ),
}


def source_client_ids(path: Path) -> set[int]:
    result = set()
    with (path / "claims_train.jsonl").open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                result.add(int(json.loads(line)["customer_id"]))
    return result


def materialize_shared_ids(
    *,
    dataset: str,
    generated_dir: Path,
    sample_size: int,
    sampling_seed: int,
    write: bool,
) -> tuple[Path, list[int]]:
    path = generated_dir / f"{dataset}_train_client_ids.json"
    qwen = source_client_ids(SOURCE_ROOTS[(dataset, "qwen")])
    gpt = source_client_ids(SOURCE_ROOTS[(dataset, "gpt_oss")])
    eligible = qwen & gpt
    base = load_yaml(Path("configs") / f"{dataset}.yaml")
    train = add_features(load_dataset(base, "train"))
    train = train[train["customer_id"].isin(eligible)].copy()
    identifiers = stratified_client_ids(
        train,
        n_clients=sample_size,
        seed=sampling_seed,
    )
    if len(identifiers) != sample_size:
        raise ValueError(
            f"{dataset}: sampled {len(identifiers)} != {sample_size}"
        )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identifiers:
            raise RuntimeError(f"Stability client IDs changed: {path}")
    elif write:
        generated_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, identifiers)
    return path, identifiers


def completed_marker(
    path: Path,
    *,
    run_id: str,
    dataset: str,
    model: str,
    seed: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("run_id") == run_id
        and payload.get("dataset") == dataset
        and payload.get("model_slug") == model
        and payload.get("generation_seed") == seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument(
        "--run-id",
        default="reviewer-v4-llm-seed-stability-v1",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 947])
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--sampling-seed", type=int, default=137)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/v2/stability/llm-seeds-v1"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    profile = load_yaml(args.model_config)
    model = slug(profile["experiment"]["model_slug"])
    generated = Path("logs/runs") / args.run_id / "generated"
    jobs: list[dict[str, Any]] = []
    for dataset in ("rosbank", "gender", "age"):
        ids_path, identifiers = materialize_shared_ids(
            dataset=dataset,
            generated_dir=generated,
            sample_size=args.sample_size,
            sampling_seed=args.sampling_seed,
            write=args.execute,
        )
        base = load_yaml(Path("configs") / f"{dataset}.yaml")
        for seed in args.seeds:
            config = build_runtime_config(
                base,
                profile,
                run_id=args.run_id,
                variant="guided_zero_shot_v4",
                label_semantics=(
                    "age_opaque" if dataset == "age" else "standard"
                ),
                sampling_seed=args.sampling_seed,
                generation_seed=seed,
                claims_seed=17,
                ml_seed=17,
                results_root=args.results_root,
                client_ids_by_split={"train": str(ids_path)},
                expected_client_counts={"train": args.sample_size},
            )
            config_path = (
                generated / f"{dataset}_{model}_seed_{seed}.yaml"
            )
            marker = (
                Path("logs/runs")
                / args.run_id
                / "completion"
                / f"{dataset}_{model}_seed_{seed}.json"
            )
            config["output"]["completion_marker"] = str(marker)
            if args.execute:
                write_runtime_config(config_path, config)
            jobs.append({
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "ids": str(ids_path),
                "ids_sha256": (
                    file_sha256(ids_path)
                    if ids_path.is_file()
                    else fingerprint(identifiers)
                ),
                "config": str(config_path),
                "output_root": config["output"]["base_dir"],
                "completion": str(marker),
            })
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "run_id": args.run_id,
        "model": model,
        "sample_size": args.sample_size,
        "sampling_seed": args.sampling_seed,
        "generation_seeds": args.seeds,
        "claims_seed": 17,
        "jobs": jobs,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    if not args.execute_api or not args.until_complete:
        raise ValueError(
            "Execution requires --execute-api and --until-complete"
        )

    state_path = Path("logs/runs") / args.run_id / model / "queue_state.json"
    state = {
        **plan,
        "state": "running",
        "completed": [],
        "failed": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(state_path, state)
    for job in jobs:
        marker = Path(job["completion"])
        if completed_marker(
            marker,
            run_id=args.run_id,
            dataset=job["dataset"],
            model=model,
            seed=job["seed"],
        ):
            state["completed"].append(
                f"{job['dataset']}:seed_{job['seed']}"
            )
            atomic_write_json(state_path, state)
            continue
        state["current"] = f"{job['dataset']}:seed_{job['seed']}"
        atomic_write_json(state_path, state)
        command = [
            sys.executable,
            "run_pipeline.py",
            "--config",
            job["config"],
            "--steps",
            "stats,prompts,cot,llm_eval,claims",
            "--splits",
            "train",
            "--execute",
            "--until-complete",
        ]
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            config = load_yaml(job["config"])
            expected = expected_ids_by_split(config, ["train"])
            evidence = validate_completion(config, ["train"], expected)
        except Exception as error:
            state["state"] = "failed"
            state["failed"] = {
                "cell": state["current"],
                "type": type(error).__name__,
                "message": str(error),
            }
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(state_path, state)
            raise
        identity = {
            "status": "completed",
            "run_id": args.run_id,
            "dataset": job["dataset"],
            "model_slug": model,
            "generation_seed": job["seed"],
            "claims_seed": 17,
            "splits": ["train"],
            "expected_counts": {"train": args.sample_size},
            "manifest_sha256": evidence["manifest_sha256"],
            "ids_sha256": job["ids_sha256"],
        }
        identity["completion_signature"] = fingerprint(identity)
        atomic_write_json(marker, identity)
        state["completed"].append(state["current"])
        atomic_write_json(state_path, state)
    state["state"] = "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()

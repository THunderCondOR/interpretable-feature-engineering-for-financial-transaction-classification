#!/usr/bin/env python3
"""Wait for isolated API cells, then build clusters and fidelity sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json
from src.experiments.config_builder import load_yaml, slug


DATASETS = ("datafusion_default_2023", "cofinfad_operational_fidelity")
MODELS = ("qwen", "gpt_oss")
ML_EXPERIMENTS = (
    "standard",
    "llm_profile",
    "standard_profile",
    "handcrafted",
    "cot",
    "concat",
    "standard_cot",
    "all_nonclaim",
    "all_features",
)


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def completion_ready(run_id: str, dataset: str, model: str) -> bool:
    path = Path("logs/runs") / run_id / "completion" / f"{model}_{dataset}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("run_id") == run_id
        and payload.get("dataset") == dataset
        and payload.get("model_slug") == model
        and bool(payload.get("completion_signature"))
    )


def run(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generated-root", type=Path, default=Path("logs/runs/isolated/generated"))
    parser.add_argument("--derived-root", type=Path, default=Path("results/isolated/derived"))
    parser.add_argument("--gpu-blocker-session", default="reviewer_v10_lora_all_datasets")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute else "dry-run", "run_id": args.run_id,
        "dataset_order": list(DATASETS), "model_order_within_dataset": list(MODELS),
        "waits_for": "both API model cells per dataset and current LoRA session",
        "clustering": {"embedding": "intfloat/multilingual-e5-large", "backend": "minibatch_kmeans", "candidates": [100, 200, 400, 800]},
        "ml_experiments": list(ML_EXPERIMENTS),
        "fidelity": {"datafusion_default_2023": "official RNN probability teacher", "cofinfad_operational_fidelity": "published continuous churn score"},
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    state_path = args.derived_root / args.run_id / "offline_queue_state.json"
    previous = json.loads(state_path.read_text()) if state_path.is_file() else {}
    completed = set(previous.get("completed", []))
    state = {**plan, "state": "running", "completed": sorted(completed)}
    atomic_write_json(state_path, state)
    for dataset in DATASETS:
        while not all(completion_ready(args.run_id, dataset, model) for model in MODELS):
            state.update(state="waiting_api", current=dataset)
            atomic_write_json(state_path, state)
            time.sleep(max(5.0, args.poll_seconds))
        while args.gpu_blocker_session and session_exists(args.gpu_blocker_session):
            state.update(state="waiting_gpu", current=dataset, blocker=args.gpu_blocker_session)
            atomic_write_json(state_path, state)
            time.sleep(max(5.0, args.poll_seconds))
        for model in MODELS:
            key = f"clusters:{dataset}:{model}"
            config_path = (
                args.generated_root / slug(args.run_id) / dataset
                / f"selected_{model}.yaml"
            )
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing selected source config: {config_path}")
            source_root = Path(load_yaml(config_path)["output"]["base_dir"])
            derived_cell = args.derived_root / args.run_id / dataset / model / "seed_17"
            if key not in completed:
                state.update(state="running_clusters", current=key)
                atomic_write_json(state_path, state)
                run([
                    "scripts/run_v4_offline_pipeline.py", "--source-root", str(source_root),
                    "--derived-root", str(args.derived_root / args.run_id),
                    "--embedding-model", "intfloat/multilingual-e5-large",
                    "--clustering-backend", "minibatch_kmeans",
                    "--candidates", "k_100", "k_200", "k_400", "k_800",
                    "--ml-experiments", *ML_EXPERIMENTS,
                    "--execute",
                ])
                completed.add(key)
                state["completed"] = sorted(completed)
                atomic_write_json(state_path, state)
            fidelity_key = f"fidelity:{dataset}:{model}"
            if fidelity_key in completed:
                continue
            state.update(state="running_fidelity", current=fidelity_key)
            atomic_write_json(state_path, state)
            if dataset == "datafusion_default_2023":
                run([
                    "scripts/run_fidelity_analysis.py",
                    "--train-features", str(derived_cell / "cot_features_train.parquet"),
                    "--val-features", str(derived_cell / "cot_features_val.parquet"),
                    "--test-features", str(derived_cell / "cot_features_test.parquet"),
                    "--teacher-selection", "results/isolated/baselines/datafusion_default_2023/fidelity_teacher_selection.json",
                    "--teacher-seed", "17", "--surrogate-seed", "17",
                    "--cluster-metadata", str(derived_cell / "cot_clusters.json"),
                    "--output-dir", str(derived_cell / "fidelity"), "--execute",
                ])
            else:
                run([
                    "scripts/run_cofinfad_continuous_fidelity.py",
                    "--manifest", "data/isolated_benchmarks/cofinfad_operational_fidelity/score_activity_stratified_7500_seed137/benchmark_manifest.json",
                    "--train-features", str(derived_cell / "cot_features_train.parquet"),
                    "--val-features", str(derived_cell / "cot_features_val.parquet"),
                    "--test-features", str(derived_cell / "cot_features_test.parquet"),
                    "--output-dir", str(derived_cell / "fidelity"), "--execute",
                ])
            completed.add(fidelity_key)
            state["completed"] = sorted(completed)
            atomic_write_json(state_path, state)
    state.update(state="completed", current=None)
    atomic_write_json(state_path, state)


if __name__ == "__main__":
    main()

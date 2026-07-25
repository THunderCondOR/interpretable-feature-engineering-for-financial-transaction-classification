#!/usr/bin/env python3
"""Run seeded ML evaluation over completed reviewer-v4 offline cells."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v4_offline_pipeline import load_source, run_ml
from src.experiments.artifacts import files_fingerprint


CELLS = (
    ("rosbank", "qwen"),
    ("gender", "qwen"),
    ("age", "qwen"),
    ("rosbank", "gpt_oss"),
    ("gender", "gpt_oss"),
    ("age", "gpt_oss"),
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def completed_ml_intact(root: Path) -> bool:
    stage_path = root / "stages" / "ml.json"
    if not stage_path.is_file():
        return False
    try:
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    outputs = [Path(value) for value in stage.get("outputs", [])]
    return (
        stage.get("state") == "completed"
        and bool(outputs)
        and all(path.is_file() for path in outputs)
        and files_fingerprint(outputs) == stage.get("output_files")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("results/v2/derived/reviewer-v4-offline-v2"),
    )
    parser.add_argument(
        "--model",
        choices=("all", "qwen", "gpt_oss"),
        default="all",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    cells = [
        (dataset, model)
        for dataset, model in CELLS
        if args.model == "all" or model == args.model
    ]
    jobs = []
    for dataset, model in cells:
        root = args.derived_root / dataset / model / "seed_17"
        source_manifest = root / "source_manifest.json"
        selected = root / "stages" / "selected_features.json"
        if not source_manifest.is_file() or not selected.is_file():
            raise FileNotFoundError(
                f"Incomplete derived feature cell: {dataset}/{model}"
            )
        source = json.loads(source_manifest.read_text(encoding="utf-8"))[
            "source_contract"
        ]
        jobs.append({
            "dataset": dataset,
            "model": model,
            "derived_cell": str(root),
            "source_root": source["source_root"],
        })
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "seeds": [17, 101, 947],
        "experiments": ["standard", "handcrafted", "cot", "concat"],
        "jobs": jobs,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return

    state_path = args.derived_root / f"ml_queue_{args.model}.json"
    state = {
        **plan,
        "state": "running",
        "completed": [],
        "failed": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(state_path, state)
    for job in jobs:
        state["current"] = f"{job['dataset']}:{job['model']}"
        atomic_json(state_path, state)
        if completed_ml_intact(Path(job["derived_cell"])):
            state["completed"].append(state["current"])
            state.setdefault("reused", []).append(state["current"])
            atomic_json(state_path, state)
            continue
        try:
            _, config = load_source(Path(job["source_root"]))
            run_ml(
                output_root=Path(job["derived_cell"]),
                source=copy.deepcopy(
                    json.loads(
                        (
                            Path(job["derived_cell"])
                            / "source_manifest.json"
                        ).read_text(encoding="utf-8")
                    )["source_contract"]
                ),
                config=config,
            )
        except Exception as error:
            state["state"] = "failed"
            state["failed"] = {
                "cell": state["current"],
                "type": type(error).__name__,
                "message": str(error),
            }
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)
            raise
        state["completed"].append(state["current"])
        atomic_json(state_path, state)
    state["state"] = "completed"
    state["current"] = None
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()

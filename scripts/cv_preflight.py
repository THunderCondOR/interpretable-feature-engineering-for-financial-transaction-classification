#!/usr/bin/env python3
"""Read-only preflight for the paid v5 benchmark queues."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.benchmark_registry import BENCHMARKS, DATA_FUSION_SPLIT_BACKEND
from src.experiments.config_builder import load_yaml
from src.pipeline.prompt_builder import validate_prompt_contract


def environment() -> tuple[str, str]:
    base = os.path.expandvars(os.environ.get("API_BASE_URL", "")).strip()
    key = os.path.expandvars(os.environ.get("API_KEY", "")).strip()
    if not base.startswith(("http://", "https://")):
        raise RuntimeError("API_BASE_URL must be a resolved HTTP(S) URL")
    if not key or key.startswith("${"):
        raise RuntimeError("API_KEY must be set and resolved")
    return base, key


def validate_git_clean() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "Paid API queues require a clean committed worktree:\n"
            + "\n".join(status.splitlines()[:20])
        )
    return revision


def validate_prepared(prepared_root: Path) -> dict:
    result = {}
    for dataset, spec in BENCHMARKS.items():
        root = prepared_root / dataset / spec.protocol
        benchmark = root / "benchmark_manifest.json"
        if not benchmark.is_file():
            raise FileNotFoundError(benchmark)
        payload = json.loads(benchmark.read_text(encoding="utf-8"))
        if payload.get("benchmark", {}).get("n_entities") != spec.n_entities:
            raise RuntimeError(f"Wrong prepared entity count for {dataset}")
        folds = []
        for fold in range(5):
            path = root / f"fold_{fold}" / "fold_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if (
                manifest.get("dataset") != dataset
                or manifest.get("protocol") != spec.protocol
                or manifest.get("fold") != fold
            ):
                raise RuntimeError(f"Incompatible fold manifest: {path}")
            if (
                dataset == "datafusion_education"
                and manifest.get("split_backend") != DATA_FUSION_SPLIT_BACKEND
            ):
                raise RuntimeError(
                    "Paid Data Fusion runs require exact public notebook folds "
                    "(KFold(5, shuffle=True, random_state=100))"
                )
            if set(manifest["ids"]["outer_train"]) & set(
                manifest["ids"]["outer_test"]
            ):
                raise RuntimeError(f"Outer split overlap: {path}")
            if set(manifest["ids"]["inner_train"]) & set(
                manifest["ids"]["inner_validation"]
            ):
                raise RuntimeError(f"Inner split overlap: {path}")
            folds.append(manifest["counts"])
        result[dataset] = folds
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("data/benchmarks_v5")
    )
    parser.add_argument(
        "--qwen-config", type=Path, default=Path("configs/v2/qwen.yaml")
    )
    parser.add_argument(
        "--gpt-config", type=Path, default=Path("configs/v2/gpt_oss.yaml")
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "checks": [
            "clean git revision", "required Python imports", "API environment",
            "two model profiles", "two benchmark manifests", "ten fold manifests",
            "English prompt contracts",
        ],
    }, indent=2))
    if not args.execute:
        return
    # Keep the paid-generation preflight scoped to dependencies used before
    # and during API collection.  Embedding and booster packages belong to the
    # later offline queue and must not prevent explanations/claims collection.
    for module in ("pandas", "pyarrow", "sklearn", "openai", "yaml"):
        importlib.import_module(module)
    environment()
    revision = validate_git_clean()
    profiles = [load_yaml(path) for path in (args.qwen_config, args.gpt_config)]
    if {
        profile["experiment"]["model_slug"] for profile in profiles
    } != {"qwen", "gpt_oss"}:
        raise RuntimeError("Expected qwen and gpt_oss model profiles")
    for dataset in BENCHMARKS:
        validate_prompt_contract(load_yaml(f"configs/v5/{dataset}.yaml"))
    prepared = validate_prepared(args.prepared_root)
    print(json.dumps({
        "status": "ready",
        "git_revision": revision,
        "prepared_counts": prepared,
    }, indent=2))


if __name__ == "__main__":
    main()

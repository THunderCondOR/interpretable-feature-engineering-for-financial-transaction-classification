#!/usr/bin/env python3
"""Validate one audited explanation→claims chain per benchmark and model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.api_canary import run_model_canary
from src.benchmarks.runtime import build_isolated_runtime_config, load_benchmark_manifest
from src.experiments.config_builder import load_yaml, slug


CELLS = (
    (
        Path("configs/datafusion_default_2023/base.yaml"),
        Path("data/isolated_benchmarks/datafusion_default_2023/stratified_60_20_20_seed137/benchmark_manifest.json"),
    ),
    (
        Path("configs/cofinfad/base.yaml"),
        Path("data/isolated_benchmarks/cofinfad_operational_fidelity/score_activity_stratified_7500_seed137/benchmark_manifest.json"),
    ),
)
PROFILES = (Path("configs/v2/qwen.yaml"), Path("configs/v2/gpt_oss.yaml"))


def first_audited_prompt(dataset: str, run_id: str, generated_root: Path) -> dict:
    path = generated_root / slug(run_id) / dataset / "prompt_audits" / "guided_zero_shot_v5" / "rendered_prompts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Materialize and audit prompts before canary: {path}")
    with path.open(encoding="utf-8") as stream:
        row = json.loads(next(line for line in stream if line.strip()))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generated-root", type=Path, default=Path("logs/runs/isolated/generated"))
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute_api and args.until_complete else "dry-run",
        "benchmarks": [load_yaml(base)["dataset"]["name"] for base, _ in CELLS],
        "models": [load_yaml(path)["experiment"]["model_slug"] for path in PROFILES],
        "requests": 2 * len(CELLS) * len(PROFILES),
        "chain_per_cell": ["audited explanation", "parsed atomic claims"],
        "writes_experimental_results": False,
        "trust_environment_proxy": False,
    }
    print(json.dumps(plan, indent=2))
    if not (args.execute_api and args.until_complete):
        return
    base_url = os.path.expandvars(os.environ.get("API_BASE_URL", "")).strip()
    api_key = os.path.expandvars(os.environ.get("API_KEY", "")).strip()
    if not base_url.startswith(("http://", "https://")) or not api_key or api_key.startswith("${"):
        raise RuntimeError("Resolved API_BASE_URL and API_KEY are required")
    client = OpenAI(
        base_url=base_url, api_key=api_key, max_retries=2,
        http_client=httpx.Client(trust_env=False, timeout=180.0),
    )
    results = []
    try:
        for base_path, manifest_path in CELLS:
            base = load_yaml(base_path)
            dataset = base["dataset"]["name"]
            manifest = load_benchmark_manifest(manifest_path, dataset)
            prompt = first_audited_prompt(dataset, args.run_id, args.generated_root)
            for profile_path in PROFILES:
                config = build_isolated_runtime_config(
                    base, load_yaml(profile_path), manifest_path=manifest_path,
                    manifest=manifest, run_id=args.run_id,
                    variant="guided_zero_shot_v5", results_root=Path("/tmp/isolated_canary"),
                    mode="pilot",
                )
                results.append(run_model_canary(config=config, prompt_record=prompt, client=client))
    finally:
        client.close()
    print(json.dumps({"status": "passed", "cells": results}, indent=2))


if __name__ == "__main__":
    main()

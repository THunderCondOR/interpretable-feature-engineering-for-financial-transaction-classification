#!/usr/bin/env python3
"""Prepare and optionally execute the paired Age/Gender stereotype audit."""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_stereotype_audit_sample import read_jsonl
from scripts.run_stereotype_judge import make_dialogue
from src.experiments.artifacts import atomic_write_json


DEFAULT_JUDGES = [
    "deepseek/deepseek-v4-pro",
    "google/gemini-3-flash-preview",
]
OPENROUTER_PRICES = {
    "deepseek/deepseek-v4-pro": {"input": 0.435, "output": 0.87},
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
}


def judge_name(model: str) -> str:
    return "openrouter_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def estimate_cost(sample: Path, models: list[str], max_tokens: int) -> dict:
    rows = read_jsonl(sample)
    input_tokens = int(sum(
        len(json.dumps(make_dialogue(row), ensure_ascii=False)) / 3.5
        for row in rows
    ))
    output_tokens = len(rows) * max_tokens
    cells, total = {}, 0.0
    for model in models:
        if model not in OPENROUTER_PRICES:
            raise ValueError(f"No reviewed OpenRouter price configured for {model}")
        price = OPENROUTER_PRICES[model]
        value = (
            input_tokens * price["input"] + output_tokens * price["output"]
        ) / 1_000_000
        cells[model] = {"estimated_usd": value, **price}
        total += value
    return {
        "rationales": len(rows),
        "expected_judgments": len(rows) * len(models),
        "estimated_input_tokens_per_judge": input_tokens,
        "maximum_output_tokens_per_judge": output_tokens,
        "models": cells,
        "estimated_total_usd": total,
        "estimate_note": "Conservative chars/3.5 input estimate; retries are excluded.",
    }


def judge_command(
    *, sample: Path, output: Path, model: str, max_concurrent: int,
    max_tokens: int, proxy_url: str,
) -> list[str]:
    return [
        sys.executable, "scripts/run_stereotype_judge.py",
        "--input", str(sample), "--output", str(output),
        "--judge-name", judge_name(model), "--model", model,
        "--api-key-env", "OPENROUTER_API_KEY",
        "--proxy-url", proxy_url,
        "--max-concurrent", str(max_concurrent),
        "--batch-size", str(max_concurrent),
        "--max-tokens", str(max_tokens),
        "--execute-api", "--until-complete",
    ]


def require_proxy(proxy_url: str) -> None:
    match = re.fullmatch(r"https?://([^:/]+):(\d+)", proxy_url)
    if not match:
        raise ValueError(f"Expected an explicit HTTP proxy host:port, got {proxy_url!r}")
    try:
        with socket.create_connection((match.group(1), int(match.group(2))), timeout=3):
            pass
    except OSError as exc:
        raise RuntimeError(f"OpenRouter proxy is not reachable at {proxy_url}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v10-stereotype-audit")
    parser.add_argument("--datasets", nargs="+", default=["gender", "age"])
    parser.add_argument("--clients-per-dataset", type=int, default=60)
    parser.add_argument("--human-items", type=int, default=40)
    parser.add_argument("--sampling-seed", type=int, default=424242)
    parser.add_argument("--openrouter-models", nargs="+", default=DEFAULT_JUDGES)
    parser.add_argument("--max-openrouter-cost-usd", type=float, default=3.0)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--sources-config", type=Path,
        default=Path("configs/v5/stereotype_audit_sources.yaml"),
    )
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "run_id": args.run_id,
        "datasets": args.datasets,
        "clients_per_dataset": args.clients_per_dataset,
        "source_models": ["qwen", "gpt_oss"],
        "expected_unique_clients": len(args.datasets) * args.clients_per_dataset,
        "expected_rationales": len(args.datasets) * args.clients_per_dataset * 2,
        "expected_judgments": len(args.datasets) * args.clients_per_dataset * 4,
        "judges": args.openrouter_models,
        "human_items": args.human_items,
        "budget_usd": args.max_openrouter_cost_usd,
        "proxy": os.environ.get("OPENROUTER_PROXY_URL", "http://127.0.0.1:5300"),
        "reference_manifest": str(args.reference_manifest)
        if args.reference_manifest else None,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if args.execute_api and not args.until_complete:
        raise ValueError("--execute-api requires --until-complete")
    if len(args.openrouter_models) != 2:
        raise ValueError("The audit requires exactly two independent judges")

    output = args.output_dir or Path("results/v2/stereotype_audit") / args.run_id
    report = args.report_dir or Path("reports") / args.run_id / "stereotype_audit"
    output.mkdir(parents=True, exist_ok=True)
    sample = output / "stereotype_samples.jsonl"
    private_key = output / "stereotype_samples.private.json"
    manifest = output / "sample_manifest.json"
    prepare = [
        sys.executable, "scripts/prepare_stereotype_audit_sample.py",
        "--sources-config", str(args.sources_config),
        "--datasets", *args.datasets,
        "--clients-per-dataset", str(args.clients_per_dataset),
        "--seed", str(args.sampling_seed),
        "--output", str(sample), "--private-key", str(private_key),
        "--manifest", str(manifest), "--execute",
    ]
    if args.reference_manifest:
        prepare.extend(["--reference-manifest", str(args.reference_manifest)])
    subprocess.run(prepare, check=True)
    annotation = [
        sys.executable, "scripts/build_stereotype_annotation.py",
        "--input", str(sample), "--private-key", str(private_key),
        "--output-html", str(report / "human_validation.html"),
        "--output-key", str(output / "human_validation.key.json"),
        "--human-items", str(args.human_items), "--execute",
    ]
    subprocess.run(annotation, check=True)

    cost = estimate_cost(sample, args.openrouter_models, args.max_tokens)
    cost["budget_usd"] = args.max_openrouter_cost_usd
    atomic_write_json(output / "openrouter_cost_estimate.json", cost)
    print(json.dumps(cost, indent=2))
    if cost["estimated_total_usd"] > args.max_openrouter_cost_usd:
        raise RuntimeError(
            f"Estimated cost ${cost['estimated_total_usd']:.2f} exceeds "
            f"budget ${args.max_openrouter_cost_usd:.2f}"
        )

    proxy_url = os.environ.get("OPENROUTER_PROXY_URL", "http://127.0.0.1:5300")
    outputs, commands = [], []
    for model in args.openrouter_models:
        result = output / f"judge_{judge_name(model)}.jsonl"
        outputs.append(result)
        commands.append(judge_command(
            sample=sample, output=result, model=model,
            max_concurrent=args.max_concurrent, max_tokens=args.max_tokens,
            proxy_url=proxy_url,
        ))
    atomic_write_json(output / "judge_commands.json", commands)
    if not args.execute_api:
        print(f"Prepared stereotype audit without API calls -> {output}")
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    require_proxy(proxy_url)
    for command in commands:
        subprocess.run([*command, "--limit", "1"], check=True)
    for command in commands:
        subprocess.run(command, check=True)
    summarize = [
        sys.executable, "scripts/summarize_stereotype_audit.py",
        "--inputs", *map(str, outputs),
        "--expected-judges", *[judge_name(model) for model in args.openrouter_models],
        "--private-key", str(private_key),
        "--output-dir", str(report), "--execute",
    ]
    subprocess.run(summarize, check=True)
    print(f"Completed stereotype audit -> {report}")


if __name__ == "__main__":
    main()

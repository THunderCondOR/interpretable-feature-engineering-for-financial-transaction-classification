"""Materialize and execute the blinded multi-judge grounding protocol."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json
from scripts.run_grounding_judge import make_dialogue, read_jsonl


OPENROUTER_PRICES = {
    # USD per one million tokens, verified against OpenRouter on 2026-07-31.
    "deepseek/deepseek-v4-pro": {"input": 0.435, "output": 0.87},
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "openai/gpt-5.5": {"input": 5.00, "output": 30.00},
}
DEFAULT_JUDGES = [
    "deepseek/deepseek-v4-pro",
    "google/gemini-3-flash-preview",
]


def judge_name(model: str) -> str:
    return "openrouter_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def build_plan(run_id: str, models: list[str], datasets: list[str]) -> dict:
    return {
        "run_id": run_id,
        "datasets": datasets,
        "source_models": ["qwen", "gpt_oss"],
        "clients": {"primary": 50, "auxiliary": 20},
        "claims_per_client": 3,
        "sampling_seed": 271828,
        "blind_fields": ["true label", "prediction", "source model"],
        "judges": models,
        "adjudication": "Codex for disagreements and unsupported cases",
        "verdicts": ["supported", "partially_supported", "unsupported", "not_verifiable", "parse_error"],
    }


def judge_command(*, sample: Path, output: Path, model: str, api_key_env: str,
                  max_concurrent: int, max_tokens: int) -> list[str]:
    return [
        sys.executable, "scripts/run_grounding_judge.py",
        "--input", str(sample), "--output", str(output),
        "--judge-name", judge_name(model), "--model", model,
        "--api-base-url", "https://openrouter.ai/api/v1",
        "--proxy-url", os.environ.get(
            "OPENROUTER_PROXY_URL", "http://127.0.0.1:5300"
        ),
        "--api-key-env", api_key_env,
        "--max-concurrent", str(max_concurrent), "--batch-size", str(max_concurrent),
        "--max-tokens", str(max_tokens),
        "--repair-max-tokens", "512",
        "--max-repair-rounds", "3",
        "--execute-api", "--until-complete",
    ]


def estimate_cost(sample: Path, models: list[str], max_tokens: int) -> dict:
    rows = read_jsonl(sample)
    # Numeric transaction summaries usually tokenize less efficiently than prose.
    # chars/3.5 is deliberately conservative relative to the observed chars/4.
    estimated_input_tokens = int(sum(
        len(json.dumps(make_dialogue(row), ensure_ascii=False)) / 3.5
        for row in rows
    ))
    estimated_output_tokens = len(rows) * max_tokens
    cells = {}
    total = 0.0
    for model in models:
        if model not in OPENROUTER_PRICES:
            raise ValueError(
                f"No reviewed price for {model}; add it to OPENROUTER_PRICES before execution"
            )
        price = OPENROUTER_PRICES[model]
        cost = (
            estimated_input_tokens * price["input"]
            + estimated_output_tokens * price["output"]
        ) / 1_000_000
        cells[model] = {"estimated_usd": cost, **price}
        total += cost
    return {
        "samples": len(rows),
        "estimated_input_tokens_per_judge": estimated_input_tokens,
        "maximum_output_tokens_per_judge": estimated_output_tokens,
        "models": cells,
        "estimated_total_usd": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v8-grounding")
    parser.add_argument("--openrouter-models", nargs="+", default=DEFAULT_JUDGES)
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-openrouter-cost-usd", type=float, default=5.0)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--sources-config", type=Path, default=Path("configs/v5/grounding_sources.yaml"))
    parser.add_argument("--datasets", nargs="+", default=[
        "rosbank", "berka", "gender", "age",
    ])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--execute-codex", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args.run_id, args.openrouter_models, args.datasets)
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        **plan,
        "key_environment_variable": args.openrouter_api_key_env,
        "budget_usd": args.max_openrouter_cost_usd,
    }, indent=2))
    if not args.execute:
        return
    if args.execute_api and not args.until_complete:
        raise ValueError("--execute-api requires --until-complete")
    if len(args.openrouter_models) != 2:
        raise ValueError("The primary protocol requires exactly two OpenRouter judges")

    output_dir = args.output_dir or Path("results/v2") / "grounding" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = output_dir / "grounding_samples.jsonl"
    prepare = [
        sys.executable, "scripts/prepare_grounding_sample.py",
        "--datasets", *args.datasets,
        "--sources-config", str(args.sources_config),
        "--clients-per-dataset", "50",
        "--clients-per-dataset-map", "gender=20", "age=20",
        "--claims-per-client", "3",
        "--output", str(sample), "--execute",
    ]
    subprocess.run(prepare, check=True)

    annotation_command = [
        sys.executable, "scripts/build_grounding_annotation.py",
        "--input", str(sample),
        "--output-html", str(output_dir / "human_grounding_audit.html"),
        "--output-key", str(output_dir / "human_grounding_audit.key.json"),
        "--per-dataset", "30", "--execute",
    ]
    atomic_write_json(output_dir / "human_annotation.command.json", annotation_command)
    subprocess.run(annotation_command, check=True)

    cost = estimate_cost(sample, args.openrouter_models, args.max_tokens)
    cost["budget_usd"] = args.max_openrouter_cost_usd
    atomic_write_json(output_dir / "openrouter_cost_estimate.json", cost)
    print(json.dumps(cost, indent=2))
    if cost["estimated_total_usd"] > args.max_openrouter_cost_usd:
        raise RuntimeError(
            f"Estimated OpenRouter cost ${cost['estimated_total_usd']:.2f} exceeds "
            f"the ${args.max_openrouter_cost_usd:.2f} budget"
        )

    outputs = []
    commands = []
    for model in args.openrouter_models:
        output = output_dir / f"judge_{judge_name(model)}.jsonl"
        outputs.append(output)
        commands.append(judge_command(
            sample=sample,
            output=output,
            model=model,
            api_key_env=args.openrouter_api_key_env,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
        ))
    atomic_write_json(output_dir / "grounding.commands.json", commands)
    if not args.execute_api:
        print(f"Grounding sample, cost estimate and commands prepared -> {output_dir}")
        return

    # A one-item request verifies the key, model ID, structured-output
    # contract and proxy path before committing to the complete paid queue.
    # The full run reuses this result through its judgment signature.
    for command in commands:
        subprocess.run([*command, "--limit", "1"], check=True)

    for command in commands:
        subprocess.run(command, check=True)

    summary_prefix = output_dir / "grounding"
    summary_command = [
        sys.executable, "scripts/summarize_grounding_judges.py",
        "--inputs", *map(str, outputs),
        "--expected-judges", *[judge_name(model) for model in args.openrouter_models],
        "--output-prefix", str(summary_prefix), "--execute",
    ]
    subprocess.run(summary_command, check=True)
    detailed_command = [
        sys.executable, "scripts/build_grounding_detailed_report.py",
        "--inputs", *map(str, outputs),
        "--output-dir", str(Path("reports") / args.run_id / "detailed"),
        "--execute",
    ]
    subprocess.run(detailed_command, check=True)

    codex_output = output_dir / "codex_adjudication.json"
    codex_command = [
        sys.executable, "scripts/run_codex_grounding_adjudicator.py",
        "--input", str(summary_prefix.with_suffix(".disagreements.json")),
        "--output", str(codex_output), "--model", "gpt-5.5",
    ]
    atomic_write_json(output_dir / "codex.command.json", codex_command)
    if args.execute_codex:
        subprocess.run([*codex_command, "--execute-codex"], check=True)
        subprocess.run([*summary_command, "--adjudication", str(codex_output)], check=True)
    print(f"Completed automated grounding judges -> {output_dir}")


if __name__ == "__main__":
    main()

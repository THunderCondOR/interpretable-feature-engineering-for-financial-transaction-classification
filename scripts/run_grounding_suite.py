"""Materialize and execute the blinded multi-judge grounding protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json


def build_plan(run_id, openrouter_model):
    return {
        "run_id": run_id,
        "datasets": ["gender", "age", "rosbank"],
        "source_models": ["qwen", "gpt_oss"],
        "clients_per_cell": 50,
        "claims_per_client": 3,
        "sampling_seed": 271828,
        "blind_fields": ["true label", "prediction", "source model"],
        "evidence": ["verified exact client stats", "train-only reference", "field semantics", "claim"],
        "judges": ["opposite local model", f"OpenRouter:{openrouter_model}"],
        "adjudication": "Codex for disagreements and unsupported cases",
        "verdicts": ["supported", "partially_supported", "unsupported", "not_verifiable", "parse_error"],
        "metrics": ["proportions", "strict consensus", "disagreement", "Cohen kappa", "client bootstrap CI"],
    }


def judge_command(*, sample, output, judge_name, model, api_base_url, api_key_env, sources=None):
    command = [
        sys.executable, "scripts/run_grounding_judge.py",
        "--input", str(sample), "--output", str(output),
        "--judge-name", judge_name, "--model", model,
        "--api-base-url", api_base_url, "--api-key-env", api_key_env,
        "--max-concurrent", "64", "--batch-size", "64",
        "--execute-api", "--until-complete",
    ]
    if sources:
        command.extend(["--source-run-names", *sources])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v2")
    parser.add_argument("--openrouter-model", default="anthropic/claude-sonnet-4")
    parser.add_argument("--openrouter-api-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--local-api-base-url", default="${API_BASE_URL}")
    parser.add_argument("--local-api-key-env", default="API_KEY")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3.5-122B-A10B")
    parser.add_argument("--gpt-model", default="Openai/Gpt-oss-120b")
    parser.add_argument("--run-roots", nargs="+", default=["results", "results/gpt_oss_120b"])
    parser.add_argument("--run-names", nargs="+", default=["qwen", "gpt_oss"])
    parser.add_argument("--datasets", nargs="+", default=["gender", "age", "rosbank"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-unverified-legacy", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args.run_id, args.openrouter_model)
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", **plan}, indent=2))
    if not args.execute:
        return
    if args.execute_api and not args.until_complete:
        raise ValueError("--execute-api requires --until-complete")
    if len(args.run_roots) != len(args.run_names):
        raise ValueError("--run-roots and --run-names must have equal lengths")
    if args.execute_api and "${" in args.local_api_base_url:
        raise ValueError("Resolve --local-api-base-url before execution")

    output_dir = args.output_dir or Path("results/v2") / "grounding" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = output_dir / "grounding_samples.jsonl"
    prepare = [
        sys.executable, "scripts/prepare_grounding_sample.py",
        "--run-roots", *args.run_roots,
        "--run-names", *args.run_names,
        "--datasets", *args.datasets,
        "--output", str(sample),
        "--execute",
    ]
    if args.allow_unverified_legacy:
        prepare.append("--allow-unverified-legacy")
    subprocess.run(prepare, check=True)

    local_qwen = output_dir / "judge_qwen.jsonl"
    local_gpt = output_dir / "judge_gpt_oss.jsonl"
    openrouter = output_dir / "judge_openrouter.jsonl"
    qwen_sources = [name for name in args.run_names if "gpt" in name.lower()]
    gpt_sources = [name for name in args.run_names if "qwen" in name.lower()]
    commands = [
        judge_command(sample=sample, output=local_qwen, judge_name="local_qwen",
            model=args.qwen_model, api_base_url=args.local_api_base_url,
            api_key_env=args.local_api_key_env, sources=qwen_sources),
        judge_command(sample=sample, output=local_gpt, judge_name="local_gpt_oss",
            model=args.gpt_model, api_base_url=args.local_api_base_url,
            api_key_env=args.local_api_key_env, sources=gpt_sources),
        judge_command(sample=sample, output=openrouter, judge_name="openrouter",
            model=args.openrouter_model, api_base_url=args.openrouter_api_base_url,
            api_key_env=args.openrouter_api_key_env),
    ]
    atomic_write_json(output_dir / "grounding.commands.json", commands)
    if not args.execute_api:
        print(f"Grounding sample and executable commands prepared -> {output_dir}")
        return

    for command in commands:
        subprocess.run(command, check=True)
    summary_prefix = output_dir / "grounding"
    subprocess.run([
        sys.executable, "scripts/summarize_grounding_judges.py",
        "--inputs", str(local_qwen), str(local_gpt), str(openrouter),
        "--output-prefix", str(summary_prefix),
        "--execute",
    ], check=True)
    tasks = {
        "protocol": "Codex adjudicates disagreements and unsupported cases only",
        "input": str(summary_prefix.with_suffix(".disagreements.json")),
        "required_output": str(output_dir / "codex_adjudication.jsonl"),
        "blind_fields": plan["blind_fields"],
    }
    atomic_write_json(output_dir / "codex_adjudication_tasks.json", tasks)
    print(f"Completed automated grounding judges -> {output_dir}")


if __name__ == "__main__":
    main()

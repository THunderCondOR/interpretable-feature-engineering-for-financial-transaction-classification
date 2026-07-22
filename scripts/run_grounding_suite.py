"""Dry-run-first blinded grounding protocol planner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def build_plan(run_id, openrouter_model):
    return {"run_id": run_id, "datasets": ["gender", "age", "rosbank"],
        "source_models": ["qwen", "gpt_oss"], "clients_per_cell": 50,
        "claims_per_client": 3, "sampling_seed": 271828,
        "blind_fields": ["true label", "prediction", "source model"],
        "evidence": ["hashed exact client stats", "train-only reference", "field semantics", "claim"],
        "judges": ["opposite local model", f"OpenRouter:{openrouter_model}"],
        "adjudication": "Codex for disagreements and unsupported cases",
        "verdicts": ["supported", "partially_supported", "unsupported", "not_verifiable", "parse_error"],
        "metrics": ["proportions", "strict consensus", "disagreement", "Cohen kappa", "client bootstrap CI"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v2")
    parser.add_argument("--openrouter-model", default="anthropic/claude-sonnet-4")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/runs/reviewer-v2/grounding"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.run_id, args.openrouter_model)
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", **plan}, indent=2))
    if not args.execute:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "grounding.plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    commands = {"sample": "python scripts/prepare_grounding_sample.py ...",
        "local_cross_judge": "python scripts/run_grounding_judge.py ... --execute-api --until-complete",
        "openrouter_judge": f"python scripts/run_grounding_judge.py --model {args.openrouter_model} ... --execute-api --until-complete",
        "summarize": "python scripts/summarize_grounding_judges.py ...",
        "codex": "codex_adjudication_tasks.jsonl"}
    (args.output_dir / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    if args.execute_api and not args.until_complete:
        raise ValueError("--execute-api requires --until-complete")


if __name__ == "__main__":
    main()

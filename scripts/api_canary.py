#!/usr/bin/env python3
"""Validate one real English explanation→claims chain for each model.

Dry-run is the default. The canary uses one validation client and a class
reference computed only from the training split. It never writes experimental
results and never prints credentials or model responses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import openai

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.aggregator import build_dataset_summary_str
from src.data.loader import add_features, load_dataset
from src.experiments.config_builder import build_runtime_config, load_yaml
from src.pipeline.claims_extractor import _behavioral_text, _parse_claim_result
from src.pipeline.explanation_gen import _is_successful, build_output_record
from src.pipeline.prompt_builder import build_prompts


def _api_environment() -> tuple[str, str]:
    base_url = os.path.expandvars(os.environ.get("API_BASE_URL", "")).strip()
    api_key = os.path.expandvars(os.environ.get("API_KEY", "")).strip()
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError("API_BASE_URL must be a resolved http(s) URL")
    if not api_key or api_key.startswith("${"):
        raise RuntimeError("API_KEY must be set and resolved")
    return base_url, api_key


def _completion_kwargs(config: dict[str, Any], section: str) -> dict[str, Any]:
    generation = config[section]
    default_max_tokens = (
        config["pipeline"].get("claims_max_tokens", 2048)
        if section == "claims_generation"
        else config["llm"].get("max_tokens", 8192)
    )
    kwargs: dict[str, Any] = {
        "model": generation["model"],
        "temperature": float(generation.get("temperature", 0.0)),
        "top_p": float(generation.get("top_p", 1.0)),
        "max_tokens": int(generation.get("max_tokens", default_max_tokens)),
    }
    if generation.get("seed") is not None:
        kwargs["seed"] = int(generation["seed"])
    if generation.get("extra_body"):
        kwargs["extra_body"] = generation["extra_body"]
    elif config["llm"].get("extra_body"):
        kwargs["extra_body"] = config["llm"]["extra_body"]
    return kwargs


def _claims_dialogue(config: dict[str, Any], explanation: str) -> list[dict[str, str]]:
    prompt_config = config["prompts"]
    base_dir = Path(prompt_config["base_dir"])
    system = (base_dir / prompt_config["claims_system"]).read_text(encoding="utf-8")
    user_template = (base_dir / prompt_config["claims_user"]).read_text(
        encoding="utf-8"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": user_template.format(COT=_behavioral_text(explanation)),
        },
    ]


def run_model_canary(
    *,
    config: dict[str, Any],
    prompt_record: dict[str, Any],
    client: Any,
    repair_attempts: int = 3,
) -> dict[str, Any]:
    explanation_error = "no attempt"
    explanation_record: dict[str, Any] | None = None
    for _ in range(repair_attempts):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_record["system_prompt"]},
                {"role": "user", "content": prompt_record["user_prompt"]},
            ],
            **_completion_kwargs(config, "generation"),
        )
        explanation_record = build_output_record(
            {
                **prompt_record,
                "sample_id": 0,
                "min_behavioral_explanation_chars": int(
                    config.get("pipeline", {}).get(
                        "min_behavioral_explanation_chars", 80
                    )
                ),
            },
            {"response": response, "execution_time": 0.0, "error": None},
            config["dataset"]["label_names"],
        )
        if _is_successful(explanation_record):
            break
        explanation_error = (
            f"{explanation_record.get('error_type')}: "
            f"{explanation_record.get('error')}"
        )
    if explanation_record is None or not _is_successful(explanation_record):
        raise RuntimeError(f"Explanation canary failed: {explanation_error}")

    forbidden = {
        str(value)
        for value in (
            list(config["dataset"]["label_names"].values())
            + list(config["dataset"].get("claim_forbidden_terms", []))
        )
    }
    claims: list[str] = []
    claim_error = "no attempt"
    for _ in range(repair_attempts):
        response = client.chat.completions.create(
            messages=_claims_dialogue(config, explanation_record["explanation"]),
            **_completion_kwargs(config, "claims_generation"),
        )
        claims, error_type, error = _parse_claim_result(
            {"response": response, "error": None},
            forbidden_labels=forbidden,
        )
        if not error_type:
            break
        claim_error = f"{error_type}: {error}"
    if not claims:
        raise RuntimeError(f"Claims canary failed: {claim_error}")

    return {
        "model_slug": config["experiment"]["model_slug"],
        "dataset": config["dataset"]["name"],
        "client_id": int(prompt_record["customer_id"]),
        "prediction_parsed": True,
        "english_rationale": True,
        "behavioral_rationale_chars": len(
            _behavioral_text(explanation_record["explanation"])
        ),
        "claims_parsed": len(claims),
        "label_semantics": config["experiment"]["label_semantics"],
        "prompt_format": config["experiment"]["variant"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        action="append",
        type=Path,
        required=True,
        help="Repeat once per model profile.",
    )
    parser.add_argument("--dataset", choices=("gender", "age", "rosbank"), default="rosbank")
    parser.add_argument("--run-id", default="reviewer-v4-english-canary")
    parser.add_argument("--repair-attempts", type=int, default=3)
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    plan = {
        "mode": (
            "execute"
            if args.execute_api and args.until_complete
            else "dry-run"
        ),
        "dataset": args.dataset,
        "models": [str(path) for path in args.model_config],
        "chain": ["final English explanation prompt", "Final parse", "claims prompt", "claims validation"],
        "writes_experiment_results": False,
    }
    print(json.dumps(plan, indent=2))
    if not (args.execute_api and args.until_complete):
        return
    if args.repair_attempts < 1:
        raise ValueError("--repair-attempts must be positive")

    base = load_yaml(Path("configs") / f"{args.dataset}.yaml")
    profile = load_yaml(args.model_config[0])
    prompt_config = build_runtime_config(
        base,
        profile,
        run_id=args.run_id,
        variant="guided_zero_shot_v4",
        label_semantics="age_opaque" if args.dataset == "age" else "standard",
    )
    train = add_features(load_dataset(prompt_config, "train"))
    validation = add_features(load_dataset(prompt_config, "val"))
    first_client = int(validation["customer_id"].min())
    target = validation[validation["customer_id"] == first_client]
    summary = build_dataset_summary_str(train, prompt_config)
    prompt_record = build_prompts(target, prompt_config, summary, "")[0]

    base_url, api_key = _api_environment()
    client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=2)
    results: list[dict[str, Any]] = []
    try:
        for model_path in args.model_config:
            model_config = build_runtime_config(
                base,
                load_yaml(model_path),
                run_id=args.run_id,
                variant="guided_zero_shot_v4",
                label_semantics=(
                    "age_opaque" if args.dataset == "age" else "standard"
                ),
            )
            results.append(
                run_model_canary(
                    config=model_config,
                    prompt_record=prompt_record,
                    client=client,
                    repair_attempts=args.repair_attempts,
                )
            )
    finally:
        client.close()
    print(json.dumps({"status": "ok", "results": results}, indent=2))


if __name__ == "__main__":
    main()

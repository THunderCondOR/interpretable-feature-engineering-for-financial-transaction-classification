"""Bounded concurrency stress test for the configured OpenAI-compatible LLM.

The test uses real pipeline prompts but never writes explanation outputs. Metrics
are checkpointed after every scenario so an interrupted test remains useful.

Examples:
    PYTHONPATH=. python scripts/stress_test_llm.py \
        --config configs/gender.yaml --split test \
        --concurrencies 8,16,32,64,96,128 --max-tokens 64 --prompt-mode short

    PYTHONPATH=. python scripts/stress_test_llm.py \
        --config configs/gender.yaml --split test \
        --concurrencies 16,32,64 --requests-per-scenario 64 \
        --max-tokens 1024 --prompt-mode full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from src.utils.async_api import batched_query


def split_path(config: dict, key: str, split: str) -> Path:
    base = Path(config["output"][key])
    return Path(config["output"]["base_dir"]) / f"{base.stem}_{split}{base.suffix}"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def make_dialogues(prompts: list[dict], count: int, prompt_mode: str) -> list[list[dict]]:
    dialogues = []
    suffix = (
        "\n\nРежим проверки API: ответь ровно одним словом: OK."
        if prompt_mode == "short"
        else ""
    )
    for idx in range(count):
        prompt = prompts[idx % len(prompts)]
        dialogues.append(
            [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["user_prompt"] + suffix},
            ]
        )
    return dialogues


def summarize_results(
    results: list[dict],
    *,
    concurrency: int,
    request_count: int,
    wall_seconds: float,
) -> dict:
    transport_errors = Counter(
        str(result.get("error_type") or "UnknownAPIError")
        for result in results
        if result.get("error")
    )
    latencies = [
        float(result.get("execution_time", 0.0))
        for result in results
        if not result.get("error")
    ]
    finish_reasons: Counter[str] = Counter()
    empty_responses = 0
    completion_tokens = []
    for result in results:
        response = result.get("response")
        if result.get("error") or response is None:
            continue
        try:
            choice = response.choices[0]
            finish_reasons[str(choice.finish_reason)] += 1
            if not (choice.message.content or "").strip():
                empty_responses += 1
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
            if tokens is not None:
                completion_tokens.append(int(tokens))
        except Exception:
            finish_reasons["invalid_response_shape"] += 1

    failed = sum(transport_errors.values())
    return {
        "concurrency": concurrency,
        "request_count": request_count,
        "wall_seconds": wall_seconds,
        "requests_per_second": request_count / wall_seconds if wall_seconds else None,
        "transport_successful": request_count - failed,
        "transport_failed": failed,
        "transport_error_types": dict(sorted(transport_errors.items())),
        "empty_responses": empty_responses,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "latency_seconds": {
            "min": min(latencies) if latencies else None,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
        },
        "completion_tokens": {
            "mean": float(np.mean(completion_tokens)) if completion_tokens else None,
            "max": max(completion_tokens) if completion_tokens else None,
        },
    }


def parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded LLM concurrency stress test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="Override llm.default_model from config.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--concurrencies", type=parse_ints, required=True)
    parser.add_argument(
        "--requests-per-scenario",
        type=int,
        default=0,
        help="Fixed request count; 0 means one wave (request count = concurrency).",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-mode", choices=("short", "full"), default="short")
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument(
        "--stop-error-rate",
        type=float,
        default=0.25,
        help="Stop before higher concurrency when final transport error rate reaches this value.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    model = args.model or config["llm"]["default_model"]
    prompts_path = split_path(config, "prompts", args.split)
    prompts = load_jsonl(prompts_path)
    if not prompts:
        raise ValueError(f"No prompts found: {prompts_path}")

    output_path = (
        Path(args.output)
        if args.output
        else Path(config["output"]["base_dir"])
        / f"llm_stress_{args.prompt_mode}_{args.max_tokens}tokens.json"
    )
    report = {
        "config": args.config,
        "model": model,
        "split": args.split,
        "prompt_mode": args.prompt_mode,
        "max_tokens": args.max_tokens,
        "thinking_enabled": bool(
            config["llm"].get("extra_body", {})
            .get("chat_template_kwargs", {})
            .get("enable_thinking", True)
        ),
        "scenarios": [],
    }

    for scenario_index, concurrency in enumerate(args.concurrencies):
        request_count = args.requests_per_scenario or concurrency
        llm_config = dict(config["llm"])
        llm_config.update(
            {
                "max_concurrent": concurrency,
                "batch_size": request_count,
                "max_tokens": args.max_tokens,
                "max_retries": 1,
                "batch_cooldown_seconds": 0.0,
                "request_cooldown_seconds": 0.0,
            }
        )
        dialogues = make_dialogues(prompts, request_count, args.prompt_mode)
        print(
            f"\n[STRESS] concurrency={concurrency} requests={request_count} "
            f"max_tokens={args.max_tokens} mode={args.prompt_mode}"
        )
        started = time.monotonic()
        results = asyncio.run(
            batched_query(dialogues, model, llm_config)
        )
        wall_seconds = time.monotonic() - started
        scenario = summarize_results(
            results,
            concurrency=concurrency,
            request_count=request_count,
            wall_seconds=wall_seconds,
        )
        report["scenarios"].append(scenario)
        atomic_write_json(output_path, report)
        print("[STRESS RESULT] " + json.dumps(scenario, ensure_ascii=False))
        print(f"[STRESS CHECKPOINT] {output_path}")

        error_rate = scenario["transport_failed"] / request_count
        if error_rate >= args.stop_error_rate:
            print(
                f"[STRESS STOP] error_rate={error_rate:.3f} reached "
                f"threshold={args.stop_error_rate:.3f}"
            )
            break
        if scenario_index + 1 < len(args.concurrencies) and args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)


if __name__ == "__main__":
    main()

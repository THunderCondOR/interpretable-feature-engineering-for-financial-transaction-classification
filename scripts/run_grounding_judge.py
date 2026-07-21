"""Judge whether atomic claims are grounded in client transaction summaries.

The judge receives only the exact client-level transaction summary supplied to
the explanation generator and one extracted atomic claim. It must return JSON:

{"verdict": "supported|partially_supported|unsupported|not_applicable",
 "confidence": 1-5,
 "evidence": "...",
 "reason": "..."}

Works with any OpenAI-compatible endpoint, including OpenRouter and local
inference servers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.async_api import batched_query


SYSTEM_PROMPT = """You are an auditor for claim-level grounding in a financial transaction study.

Your task: decide whether the CLAIM is supported by the CLIENT TRANSACTION SUMMARY.

Use only the provided client summary. Do not use demographic stereotypes, common sense,
or external knowledge. A claim is:
- supported: directly entailed by the summary, including approximate paraphrases of counts,
  amounts, categories, frequency, income/expense totals, or explicitly listed absences.
- partially_supported: related to the summary but stronger, broader, or more interpretive
  than what the summary strictly shows.
- unsupported: not present, contradicted, or based on stereotypes/inferences not grounded
  in the summary.
- not_applicable: the claim is empty, malformed, or cannot be evaluated as behavior.

Return only valid JSON with keys: verdict, confidence, evidence, reason.
Keep evidence and reason short."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing = {}
    for record in read_jsonl(path):
        existing[str(record["sample_id"])] = record
    return existing


def make_dialogue(record: dict[str, Any]) -> list[dict[str, str]]:
    user = f"""CLIENT TRANSACTION SUMMARY:
{record["client_stats"]}

CLAIM:
{record["claim"]}

Return JSON only."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_content(result: dict[str, Any]) -> str:
    response = result.get("response")
    if response is None:
        return ""
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = text.strip()
    if not text:
        return None, "empty_response"
    try:
        return json.loads(text), None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0)), None
        except Exception as exc:
            return None, f"json_parse_error:{type(exc).__name__}"
    return None, "json_not_found"


def normalize_verdict(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "support": "supported",
        "yes": "supported",
        "grounded": "supported",
        "partially": "partially_supported",
        "partial": "partially_supported",
        "partly_supported": "partially_supported",
        "not_supported": "unsupported",
        "no": "unsupported",
        "ungrounded": "unsupported",
        "not_applicable": "not_applicable",
        "na": "not_applicable",
        "n/a": "not_applicable",
    }
    text = aliases.get(text, text)
    if text not in {"supported", "partially_supported", "unsupported", "not_applicable"}:
        return "parse_error"
    return text


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_key = args.api_key or (os.environ.get(args.api_key_env) if args.api_key_env else None)
    if not api_key:
        raise RuntimeError("Missing API key. Pass --api-key or --api-key-env.")

    samples = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    existing = load_existing(args.output)
    pending = [record for record in samples if str(record["sample_id"]) not in existing]
    print(f"judge={args.judge_name} total={len(samples)} existing={len(existing)} pending={len(pending)}")
    if not pending:
        return

    dialogues = [make_dialogue(record) for record in pending]
    llm_config = {
        "api_base_url": args.api_base_url,
        "api_key": api_key,
        "max_concurrent": args.max_concurrent,
        "http_max_connections": args.max_concurrent,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "max_retries": 5,
        "retry_backoff": 2.0,
        "rate_limit_fallback_concurrent": max(1, min(4, args.max_concurrent)),
        "rate_limit_recovery_batches": 3,
        "raise_on_error": False,
        "log_errors": True,
        "log_retries": True,
    }

    all_records = list(existing.values())

    def checkpoint(batch_results: list[tuple[int, dict[str, Any]]]) -> None:
        nonlocal all_records
        for idx, result in batch_results:
            sample = pending[idx]
            content = extract_content(result)
            parsed, parse_error = parse_json_object(content)
            verdict = normalize_verdict(parsed.get("verdict") if parsed else None)
            all_records.append(
                {
                    **sample,
                    "judge_name": args.judge_name,
                    "judge_model": args.model,
                    "verdict": verdict,
                    "confidence": parsed.get("confidence") if parsed else None,
                    "evidence": parsed.get("evidence") if parsed else "",
                    "reason": parsed.get("reason") if parsed else "",
                    "raw_response": content,
                    "parse_error": parse_error,
                    "transport_error": result.get("error"),
                    "transport_error_type": result.get("error_type"),
                    "execution_time": result.get("execution_time"),
                }
            )
        write_records(args.output, all_records)
        counts = Counter(r["verdict"] for r in all_records)
        print(f"checkpoint saved={len(all_records)} verdicts={dict(counts)} -> {args.output}")

    await batched_query(dialogues, args.model, llm_config, on_batch_complete=checkpoint)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

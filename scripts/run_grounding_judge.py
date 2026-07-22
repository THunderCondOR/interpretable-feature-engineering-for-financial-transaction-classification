"""Judge whether atomic claims are grounded in client transaction summaries.

The judge receives only the exact client-level transaction summary supplied to
the explanation generator and one extracted atomic claim. It must return JSON:

{"verdict": "supported|partially_supported|unsupported|not_verifiable",
 "confidence": 1-5,
 "evidence": "...",
 "reason": "..."}

Works with any OpenAI-compatible endpoint, including OpenRouter and local
inference servers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from src.experiments.artifacts import fingerprint


GROUNDING_PROTOCOL_VERSION = 2


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
- not_verifiable: the supplied evidence is insufficient to verify the claim.

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
        sample_id = str(record["sample_id"])
        if sample_id in existing:
            raise ValueError(f"Duplicate grounding result sample_id: {sample_id}")
        existing[sample_id] = record
    return existing


def make_dialogue(record: dict[str, Any]) -> list[dict[str, str]]:
    user = f"""CLIENT TRANSACTION SUMMARY:
{record["client_stats"]}

TRAIN-ONLY REFERENCE SUMMARY:
{record.get("train_reference_summary", "not supplied")}

FIELD SEMANTICS:
{record.get("field_semantics", "not supplied")}

CLAIM:
{record["claim"]}

Return JSON only."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def judgment_signature(record: dict[str, Any], judge_signature: str) -> str:
    """Bind reuse to the complete rendered evidence, claim and protocol."""
    return fingerprint({
        "protocol_version": GROUNDING_PROTOCOL_VERSION,
        "judge_signature": judge_signature,
        "evidence_hash": record.get("evidence_hash"),
        "claim": record.get("claim"),
        "dialogue": make_dialogue(record),
    })


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
        "not_applicable": "not_verifiable",
        "not_verifiable": "not_verifiable",
        "na": "not_verifiable",
        "n/a": "not_verifiable",
    }
    text = aliases.get(text, text)
    if text not in {"supported", "partially_supported", "unsupported", "not_verifiable"}:
        return "parse_error"
    return text


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-concurrent", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-run-names", nargs="+")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    if not (args.execute_api and args.until_complete):
        print(json.dumps({"mode": "dry-run", "judge": args.judge_name,
            "model": args.model, "requires": ["--execute-api", "--until-complete"]}, indent=2))
        return

    api_key = args.api_key or (os.environ.get(args.api_key_env) if args.api_key_env else None)
    if not api_key:
        raise RuntimeError("Missing API key. Pass --api-key or --api-key-env.")

    samples = read_jsonl(args.input)
    if args.source_run_names:
        allowed_sources = set(args.source_run_names)
        samples = [row for row in samples if row.get("run_name") in allowed_sources]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    judge_signature = hashlib.sha256(json.dumps({
        "protocol_version": GROUNDING_PROTOCOL_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "judge": args.judge_name,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }, sort_keys=True).encode()).hexdigest()
    loaded_existing = load_existing(args.output)
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("Duplicate grounding input sample_id")
    existing = {
        sample_id: record
        for sample_id, record in loaded_existing.items()
        if record.get("judge_signature") == judge_signature
        and sample_id in sample_by_id
        and record.get("judgment_signature")
        == judgment_signature(sample_by_id[sample_id], judge_signature)
    }
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
        "rate_limit_recovery_batches": 10,
        "initial_concurrency": args.max_concurrent,
        "fallback_concurrency": 10,
        "recovery_clean_batches": 10,
        "cooldown_seconds": 60,
        "raise_on_error": False,
        "log_errors": True,
        "log_retries": True,
        "until_complete": True,
        "request_keys": [str(record["sample_id"]) for record in pending],
        "generation_signature": hashlib.sha256(
            json.dumps({
                "judge_signature": judge_signature,
                "requests": [
                    judgment_signature(record, judge_signature) for record in pending
                ],
            }, sort_keys=True).encode()
        ).hexdigest(),
        "scheduler_state_dir": str(args.output.parent / ".scheduler" / args.output.stem),
        "events_path": str(args.output.parent / ".scheduler" / f"{args.output.stem}.events.jsonl"),
    }

    all_records = list(existing.values())

    def checkpoint(batch_results: list[tuple[int, dict[str, Any]]]) -> None:
        nonlocal all_records
        staged = []
        for idx, result in batch_results:
            sample = pending[idx]
            content = extract_content(result)
            parsed, parse_error = parse_json_object(content)
            verdict = normalize_verdict(parsed.get("verdict") if parsed else None)
            staged.append({
                    **sample,
                    "judge_name": args.judge_name,
                    "judge_model": args.model,
                    "judge_signature": judge_signature,
                    "grounding_protocol_version": GROUNDING_PROTOCOL_VERSION,
                    "judgment_signature": judgment_signature(sample, judge_signature),
                    "verdict": verdict,
                    "confidence": parsed.get("confidence") if parsed else None,
                    "evidence": parsed.get("evidence") if parsed else "",
                    "reason": parsed.get("reason") if parsed else "",
                    "raw_response": content,
                    "parse_error": parse_error,
                    "transport_error": result.get("error"),
                    "transport_error_type": result.get("error_type"),
                    "execution_time": result.get("execution_time"),
                })
        invalid = [row for row in staged if row["verdict"] == "parse_error" or row["transport_error"]]
        if invalid:
            raise RuntimeError(f"repairable grounding window errors: {len(invalid)}")
        all_records.extend(staged)
        write_records(args.output, all_records)
        counts = Counter(r["verdict"] for r in all_records)
        print(f"checkpoint saved={len(all_records)} verdicts={dict(counts)} -> {args.output}")

    await batched_query(dialogues, args.model, llm_config, on_batch_complete=checkpoint)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

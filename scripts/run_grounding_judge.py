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


GROUNDING_PROTOCOL_VERSION = 4

CLAIM_TYPES = {
    "direct_observation",
    "train_relative_comparison",
    "temporal_or_activity_interpretation",
    "higher_level_behavioral_interpretation",
    "demographic_or_social_inference",
}


SYSTEM_PROMPT = """You are an auditor for claim-level grounding in a financial transaction study.

Decide whether the CLAIM is supported by the supplied evidence. Treat the CLIENT
TRANSACTION SUMMARY as the primary evidence. Use the TRAIN-ONLY REFERENCE SUMMARY
only to verify explicit comparative claims. Always follow FIELD SEMANTICS; never
assign income, expense, inflow, or outflow meaning that the field semantics do not
support.

Do not use demographic stereotypes or unrelated outside assumptions as evidence.
Limited interpretation is allowed only to assess whether a hypothesis has a
relevant transaction-based basis. A verdict is:
- supported: directly entailed by the client evidence, including an accurate
  qualitative paraphrase of an explicitly supplied value, category, or trend;
- partially_supported: a plausible interpretation that has relevant transaction
  evidence but goes beyond what the evidence directly establishes;
- unsupported: contradicted by the evidence or asserted without any relevant
  transaction evidence;
- not_verifiable: the necessary field is genuinely omitted, truncated, or
  semantically ambiguous, so the claim cannot be assessed.

Also classify the CLAIM itself with exactly one claim_type:
- direct_observation;
- train_relative_comparison;
- temporal_or_activity_interpretation;
- higher_level_behavioral_interpretation;
- demographic_or_social_inference.

Return only valid JSON with keys verdict, claim_type, confidence, evidence, and
reason. confidence must be an integer from 1 to 5. Keep evidence and reason
short."""


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
    # Put the dataset-level prefix before client-specific evidence. This keeps
    # the semantics unchanged while allowing providers to cache the repeated
    # prefix across claims from the same dataset.
    user = f"""TRAIN-ONLY REFERENCE SUMMARY:
{record.get("train_reference_summary", "not supplied")}

FIELD SEMANTICS:
{record.get("field_semantics", "not supplied")}

CLIENT TRANSACTION SUMMARY:
{record["client_stats"]}

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


def _coerce_json_object(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Accept an object or an unambiguous provider-wrapped object."""
    if isinstance(value, dict):
        return value, None
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0], None
    return None, f"json_not_object:{type(value).__name__}"


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = text.strip()
    if not text:
        return None, "empty_response"
    try:
        return _coerce_json_object(json.loads(text))
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return _coerce_json_object(json.loads(match.group(0)))
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


def validate_judgment(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return "judgment_not_an_object"
    missing = {"verdict", "claim_type", "confidence", "evidence", "reason"} - set(payload)
    if missing:
        return "missing_keys:" + ",".join(sorted(missing))
    confidence = payload.get("confidence")
    if not isinstance(confidence, int) or isinstance(confidence, bool):
        return "confidence_not_integer"
    if not 1 <= confidence <= 5:
        return "confidence_out_of_range"
    if not isinstance(payload.get("evidence"), str) or not isinstance(
        payload.get("reason"), str
    ):
        return "evidence_or_reason_not_string"
    if normalize_verdict(payload.get("verdict")) == "parse_error":
        return "invalid_verdict"
    if payload.get("claim_type") not in CLAIM_TYPES:
        return "invalid_claim_type"
    return None


def response_usage(result: dict[str, Any]) -> dict[str, int | float | None]:
    """Extract portable usage fields from an OpenAI-compatible response."""
    response = result.get("response")
    usage = getattr(response, "usage", None) if response is not None else None
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_repair_attempts(path: Path, records: list[dict[str, Any]]) -> None:
    """Keep failed responses for diagnosis without treating them as complete."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument(
        "--proxy-url",
        default=os.environ.get("OPENROUTER_PROXY_URL", "http://127.0.0.1:5300"),
        help="Explicit proxy used for OpenRouter requests",
    )
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-concurrent", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--repair-max-tokens", type=int, default=512)
    parser.add_argument("--max-repair-rounds", type=int, default=3)
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
        "reasoning_enabled": False,
        "response_format": "json_object",
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
        unresolved_path = args.output.with_name(
            args.output.stem + ".unresolved.jsonl"
        )
        if unresolved_path.exists():
            unresolved_path.unlink()
        return

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
        # Grounding judgments are independent. A malformed answer must not
        # discard valid paid answers from the same transport batch.
        "atomic_windows": False,
        "proxy_url": args.proxy_url,
        "use_env_proxy": False,
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
        # Grounding is a short evidence-classification task. Disabling hidden
        # reasoning keeps frontier judges within the declared token budget and
        # avoids consuming the JSON output allowance with reasoning tokens.
        "extra_body": {
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_object"},
        },
    }

    all_records = dict(existing)
    repair_path = args.output.with_name(args.output.stem + ".repair_attempts.jsonl")
    repair_round = 0

    while pending:
        repair_round += 1
        if repair_round > args.max_repair_rounds + 1:
            unresolved_path = args.output.with_name(
                args.output.stem + ".unresolved.jsonl"
            )
            write_records(unresolved_path, pending)
            raise RuntimeError(
                f"Grounding repair exhausted for {len(pending)} judgments; "
                f"saved unresolved items to {unresolved_path}"
            )
        dialogues = [make_dialogue(record) for record in pending]
        # The first pass keeps the reviewed cost estimate. Only malformed or
        # truncated answers get a larger response allowance on repair passes.
        llm_config["max_tokens"] = (
            args.max_tokens if repair_round == 1 else args.repair_max_tokens
        )
        print(
            f"judge={args.judge_name} repair_round={repair_round} "
            f"pending={len(pending)} saved={len(all_records)}"
        )

        def checkpoint(batch_results: list[tuple[int, dict[str, Any]]]) -> None:
            staged_valid = []
            staged_invalid = []
            for idx, result in batch_results:
                sample = pending[idx]
                content = extract_content(result)
                parsed, parse_error = parse_json_object(content)
                validation_error = validate_judgment(parsed)
                if not parse_error and validation_error:
                    parse_error = validation_error
                verdict = normalize_verdict(parsed.get("verdict") if parsed else None)
                if parse_error:
                    verdict = "parse_error"
                row = {
                    **sample,
                    "judge_name": args.judge_name,
                    "judge_model": args.model,
                    "judge_signature": judge_signature,
                    "grounding_protocol_version": GROUNDING_PROTOCOL_VERSION,
                    "judgment_signature": judgment_signature(sample, judge_signature),
                    "verdict": verdict,
                    "claim_type": parsed.get("claim_type") if parsed else None,
                    "confidence": parsed.get("confidence") if parsed else None,
                    "evidence": parsed.get("evidence") if parsed else "",
                    "reason": parsed.get("reason") if parsed else "",
                    "raw_response": content,
                    "parse_error": parse_error,
                    "transport_error": result.get("error"),
                    "transport_error_type": result.get("error_type"),
                    "execution_time": result.get("execution_time"),
                    "usage": response_usage(result),
                    "response_max_tokens": llm_config["max_tokens"],
                    "repair_round": repair_round,
                }
                if row["verdict"] == "parse_error" or row["transport_error"]:
                    staged_invalid.append(row)
                else:
                    staged_valid.append(row)

            for row in staged_valid:
                all_records[str(row["sample_id"])] = row
            if staged_valid:
                write_records(
                    args.output,
                    [all_records[str(sample["sample_id"])] for sample in samples
                     if str(sample["sample_id"]) in all_records],
                )
            append_repair_attempts(repair_path, staged_invalid)
            counts = Counter(r["verdict"] for r in all_records.values())
            error_counts = Counter(
                (r.get("transport_error_type") or r.get("parse_error") or "unknown")
                for r in staged_invalid
            )
            print(
                f"checkpoint saved={len(all_records)} valid_in_batch={len(staged_valid)} "
                f"repair_in_batch={len(staged_invalid)} errors={dict(error_counts)} "
                f"verdicts={dict(counts)} -> {args.output}"
            )

        await batched_query(
            dialogues, args.model, llm_config, on_batch_complete=checkpoint
        )
        pending = [
            record for record in samples
            if str(record["sample_id"]) not in all_records
        ]
        if pending:
            print(
                f"repair queue: {len(pending)} judgments remain after round "
                f"{repair_round}; valid results are already durable"
            )

    # A later resume can complete a previously exhausted repair queue.  Do not
    # leave the old unresolved marker behind once exact coverage is restored.
    unresolved_path = args.output.with_name(
        args.output.stem + ".unresolved.jsonl"
    )
    if unresolved_path.exists():
        unresolved_path.unlink()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

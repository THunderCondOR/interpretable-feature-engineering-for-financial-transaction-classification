#!/usr/bin/env python3
"""Run GPT-5.5 as a blinded third grounding judge over every sample.

The Codex process receives a sanitised chunk containing only the claim and the
three permitted evidence fields.  Existing OpenRouter/local-judge artifacts are
never read or modified by this runner.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json, fingerprint
from scripts.run_grounding_judge import CLAIM_TYPES, GROUNDING_PROTOCOL_VERSION


JUDGE_NAME = "codex_gpt_5_5_full"
MODEL = "gpt-5.5"
FULL_JUDGE_PROTOCOL_VERSION = 1
VERDICTS = (
    "supported",
    "partially_supported",
    "unsupported",
    "not_verifiable",
)
BLINDED_FIELDS = (
    "claim",
    "client_stats",
    "train_reference_summary",
    "field_semantics",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_samples(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    required = set(BLINDED_FIELDS) | {"evidence_hash"}
    ids: list[str] = []
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"Grounding sample {row.get('sample_id')} lacks {sorted(missing)}"
            )
        ids.append(str(row["sample_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate grounding sample_id")
    return rows


def opaque_task_id(sample: dict[str, Any]) -> str:
    """Hide source-model-bearing sample IDs from the external judge."""
    return "task_" + fingerprint({
        "protocol": FULL_JUDGE_PROTOCOL_VERSION,
        "sample_id": str(sample["sample_id"]),
        "evidence_hash": sample["evidence_hash"],
    })[:24]


def blinded_task(sample: dict[str, Any]) -> dict[str, Any]:
    """Return the complete and exclusive payload visible to Codex."""
    return {
        "task_id": opaque_task_id(sample),
        **{field: sample[field] for field in BLINDED_FIELDS},
    }


def output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["judgments"],
        "properties": {
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "task_id",
                        "verdict",
                        "claim_type",
                        "confidence",
                        "evidence",
                        "reason",
                    ],
                    "properties": {
                        "task_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "claim_type": {
                            "type": "string",
                            "enum": sorted(CLAIM_TYPES),
                        },
                        "confidence": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                        },
                        "evidence": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            }
        },
    }


def build_prompt(tasks_path: Path) -> str:
    return f"""You are the third independent judge in a claim-level grounding audit.
Read only the JSON tasks in {tasks_path.resolve()}. Do not inspect any other files
or use outside facts. For every task, assess CLAIM using only client_stats as
primary evidence, train_reference_summary only for explicit relative comparisons,
and field_semantics to interpret fields. Do not infer labels, predictions, source
models, demographic stereotypes, or facts absent from the transaction evidence.

Verdicts:
- supported: directly entailed or an accurate qualitative paraphrase;
- partially_supported: has relevant transaction evidence but extends beyond it;
- unsupported: contradicted or has no relevant transaction evidence;
- not_verifiable: required evidence is genuinely omitted or ambiguous.

Classify claim_type as one of: {', '.join(sorted(CLAIM_TYPES))}.
Return every task_id exactly once with verdict, claim_type, integer confidence
1-5, concise evidence, and concise reason. Return only the requested JSON."""


def judge_signature() -> str:
    return fingerprint({
        "full_judge_protocol_version": FULL_JUDGE_PROTOCOL_VERSION,
        "grounding_protocol_version": GROUNDING_PROTOCOL_VERSION,
        "judge_name": JUDGE_NAME,
        "judge_model": MODEL,
        "schema": output_schema(),
    })


def judgment_signature(sample: dict[str, Any]) -> str:
    return fingerprint({
        "judge_signature": judge_signature(),
        "evidence_hash": sample["evidence_hash"],
        "task": blinded_task(sample),
    })


def validate_chunk(
    expected: list[dict[str, Any]], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    validated = validate_partial_chunk(expected, payload)
    if len(validated) != len(expected):
        raise ValueError("Codex chunk sample IDs are incomplete or unexpected")
    return validated


def validate_partial_chunk(
    expected: list[dict[str, Any]], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate and preserve the valid subset returned by an incomplete call."""
    rows = payload.get("judgments")
    if not isinstance(rows, list):
        raise ValueError("Codex output lacks judgments array")
    expected_by_id = {opaque_task_id(row): row for row in expected}
    if len(expected_by_id) != len(expected):
        raise ValueError("Opaque full-judge task ID collision")
    observed_ids = [str(row.get("task_id")) for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Duplicate Codex sample_id in chunk")
    unexpected = set(observed_ids) - set(expected_by_id)
    if unexpected:
        raise ValueError("Codex chunk contains unexpected sample IDs")
    validated = []
    for row in rows:
        task_id = str(row["task_id"])
        sample = expected_by_id[task_id]
        sample_id = str(sample["sample_id"])
        if row.get("verdict") not in VERDICTS:
            raise ValueError(f"Invalid verdict for {sample_id}")
        if row.get("claim_type") not in CLAIM_TYPES:
            raise ValueError(f"Invalid claim_type for {sample_id}")
        confidence = row.get("confidence")
        if (
            not isinstance(confidence, int)
            or isinstance(confidence, bool)
            or not 1 <= confidence <= 5
        ):
            raise ValueError(f"Invalid confidence for {sample_id}")
        for field in ("evidence", "reason"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Invalid {field} for {sample_id}")
        validated.append({
            **sample,
            "sample_id": sample_id,
            "judge_name": JUDGE_NAME,
            "judge_model": MODEL,
            "judge_signature": judge_signature(),
            "grounding_protocol_version": GROUNDING_PROTOCOL_VERSION,
            "full_judge_protocol_version": FULL_JUDGE_PROTOCOL_VERSION,
            "judgment_signature": judgment_signature(sample),
            "verdict": row["verdict"],
            "claim_type": row["claim_type"],
            "confidence": confidence,
            "evidence": row["evidence"].strip(),
            "reason": row["reason"].strip(),
            "parse_error": None,
            "transport_error": None,
        })
    return validated


def raw_judgment(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a validated judgment back to the canonical schema payload."""
    return {
        "task_id": opaque_task_id(row),
        "verdict": row["verdict"],
        "claim_type": row["claim_type"],
        "confidence": int(row["confidence"]),
        "evidence": row["evidence"],
        "reason": row["reason"],
    }


def run_codex_chunk(
    *, expected: list[dict[str, Any]], tasks_path: Path,
    response_path: Path, schema_path: Path,
) -> list[dict[str, Any]]:
    atomic_write_json(tasks_path, [blinded_task(row) for row in expected])
    command = [
        "codex", "exec", "--ephemeral", "--sandbox", "read-only",
        "--cd", str(REPO_ROOT), "--model", MODEL,
        "--output-schema", str(schema_path.resolve()),
        "--output-last-message", str(response_path.resolve()),
        build_prompt(tasks_path),
    ]
    subprocess.run(command, check=True)
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    return validate_partial_chunk(expected, payload)


def repair_incomplete_chunk(
    *, expected: list[dict[str, Any]], tasks_path: Path,
    response_path: Path, schema_path: Path, max_attempts: int,
) -> list[dict[str, Any]]:
    """Salvage valid rows and retry only missing task IDs."""
    accepted: dict[str, dict[str, Any]] = {}
    if response_path.exists():
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
            for row in validate_partial_chunk(expected, payload):
                accepted[str(row["sample_id"])] = row
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(
                "Ignoring unusable incomplete response "
                f"{response_path}: {type(error).__name__}: {error}",
                flush=True,
            )
    expected_by_sample = {str(row["sample_id"]): row for row in expected}
    for attempt in range(1, max_attempts + 1):
        missing = [
            row for sample_id, row in expected_by_sample.items()
            if sample_id not in accepted
        ]
        if not missing:
            break
        repair_tasks = tasks_path.with_name(
            f"{tasks_path.stem}.repair_{attempt:02d}{tasks_path.suffix}"
        )
        repair_response = response_path.with_name(
            f"{response_path.stem}.repair_{attempt:02d}{response_path.suffix}"
        )
        print(
            f"Repair attempt={attempt}/{max_attempts} missing={len(missing)} "
            f"preserved={len(accepted)}",
            flush=True,
        )
        rows = run_codex_chunk(
            expected=missing, tasks_path=repair_tasks,
            response_path=repair_response, schema_path=schema_path,
        )
        for row in rows:
            accepted[str(row["sample_id"])] = row
    missing_ids = set(expected_by_sample) - set(accepted)
    if missing_ids:
        raise ValueError(
            f"Codex chunk remains incomplete after repairs: missing={len(missing_ids)}"
        )
    ordered = [accepted[str(row["sample_id"])] for row in expected]
    atomic_write_json(response_path, {"judgments": [raw_judgment(row) for row in ordered]})
    return ordered


def load_resumed_chunk(
    *,
    expected: list[dict[str, Any]],
    response_path: Path,
    resume_path: Path,
    chunk_signature: str,
    start: int,
) -> list[dict[str, Any]] | None:
    if not response_path.exists() or not resume_path.exists():
        return None
    try:
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
        if resume.get("chunk_signature") != chunk_signature:
            return None
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        rows = validate_chunk(expected, payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(
            f"Rejecting corrupt full-judge resume at offset {start}: "
            f"{type(error).__name__}: {error}"
        )
        return None
    print(f"Reusing validated full-judge chunk at offset {start}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_complete(
    samples: list[dict[str, Any]], judgments: list[dict[str, Any]]
) -> None:
    expected = {str(row["sample_id"]): row for row in samples}
    observed_ids = [str(row["sample_id"]) for row in judgments]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(expected):
        raise ValueError("Full Codex judge output does not exactly cover input IDs")
    for row in judgments:
        sample = expected[str(row["sample_id"])]
        if row.get("evidence_hash") != sample.get("evidence_hash"):
            raise ValueError(f"Evidence hash mismatch for {row['sample_id']}")
        if row.get("judgment_signature") != judgment_signature(sample):
            raise ValueError(f"Judgment signature mismatch for {row['sample_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=MODEL, choices=[MODEL])
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--max-repair-attempts", type=int, default=5)
    parser.add_argument("--execute-codex", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.max_repair_attempts <= 0:
        raise ValueError("chunk-size and max-repair-attempts must be positive")

    samples = load_samples(args.input)
    plan = {
        "mode": "execute" if args.execute_codex else "dry-run",
        "judge_name": JUDGE_NAME,
        "judge_model": MODEL,
        "samples": len(samples),
        "chunk_size": args.chunk_size,
        "output": str(args.output),
        "codex_sandbox": "read-only",
        "visible_fields": ["task_id", *BLINDED_FIELDS],
    }
    print(json.dumps(plan, indent=2))
    if not args.execute_codex:
        return

    work = args.output.parent / f".{args.output.stem}.chunks"
    work.mkdir(parents=True, exist_ok=True)
    schema_path = work / "full_judge.schema.json"
    atomic_write_json(schema_path, output_schema())
    completed: dict[str, dict[str, Any]] = {}
    for start in range(0, len(samples), args.chunk_size):
        original_chunk = samples[start:start + args.chunk_size]
        visible_chunk = [blinded_task(row) for row in original_chunk]
        tasks_path = work / f"tasks_{start:05d}.json"
        response_path = work / f"response_{start:05d}.json"
        resume_path = work / f"response_{start:05d}.resume.json"
        atomic_write_json(tasks_path, visible_chunk)
        chunk_signature = fingerprint({
            "judge_signature": judge_signature(),
            "tasks": visible_chunk,
            "evidence_hashes": [row["evidence_hash"] for row in original_chunk],
        })
        rows = load_resumed_chunk(
            expected=original_chunk,
            response_path=response_path,
            resume_path=resume_path,
            chunk_signature=chunk_signature,
            start=start,
        )
        if rows is None:
            rows = repair_incomplete_chunk(
                expected=original_chunk, tasks_path=tasks_path,
                response_path=response_path, schema_path=schema_path,
                max_attempts=args.max_repair_attempts,
            )
            atomic_write_json(resume_path, {"chunk_signature": chunk_signature})
        for row in rows:
            completed[str(row["sample_id"])] = row
        ordered = [
            completed[str(sample["sample_id"])]
            for sample in samples
            if str(sample["sample_id"]) in completed
        ]
        write_jsonl(args.output, ordered)
        print(f"checkpoint completed={len(ordered)}/{len(samples)} -> {args.output}")

    judgments = read_jsonl(args.output)
    validate_complete(samples, judgments)
    print(f"Saved {len(judgments)} complete GPT-5.5 judgments -> {args.output}")


if __name__ == "__main__":
    main()

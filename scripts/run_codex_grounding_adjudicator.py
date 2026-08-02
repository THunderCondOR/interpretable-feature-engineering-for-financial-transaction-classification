#!/usr/bin/env python3
"""Adjudicate grounding disagreements with isolated structured Codex runs."""
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

VERDICTS = [
    "supported",
    "partially_supported",
    "unsupported",
    "not_verifiable",
]
ADJUDICATION_PROTOCOL_VERSION = 2
BLINDED_FIELDS = (
    "claim",
    "client_stats",
    "train_reference_summary",
    "field_semantics",
)


def opaque_task_id(task: dict[str, Any]) -> str:
    return "task_" + fingerprint({
        "protocol": ADJUDICATION_PROTOCOL_VERSION,
        "sample_id": str(task["sample_id"]),
        "evidence_hash": task.get("evidence_hash"),
    })[:24]


def blinded_task(task: dict[str, Any]) -> dict[str, Any]:
    """Projection visible to Codex; source/judge metadata stay host-side."""
    return {
        "task_id": opaque_task_id(task),
        **{field: task[field] for field in BLINDED_FIELDS},
    }


def load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Codex adjudication input must be a JSON array")
    required = {
        "sample_id",
        "claim",
        "client_stats",
        "train_reference_summary",
        "field_semantics",
        "judge_details",
    }
    for row in payload:
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"Adjudication task {row.get('sample_id')} lacks {sorted(missing)}"
            )
    return payload


def output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["adjudications"],
        "properties": {
            "adjudications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "task_id",
                        "verdict",
                        "confidence",
                        "reason",
                    ],
                    "properties": {
                        "task_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": VERDICTS},
                        "confidence": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                        },
                        "reason": {"type": "string"},
                    },
                },
            }
        },
    }


def validate_chunk(
    expected: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    model: str,
) -> list[dict[str, Any]]:
    rows = payload.get("adjudications")
    if not isinstance(rows, list):
        raise ValueError("Codex output lacks adjudications array")
    expected_by_id = {opaque_task_id(row): row for row in expected}
    if len(expected_by_id) != len(expected):
        raise ValueError("Opaque adjudication task ID collision")
    observed_ids = [str(row.get("task_id")) for row in rows]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != set(expected_by_id):
        raise ValueError("Codex adjudication sample IDs are incomplete or duplicated")
    for row in rows:
        if row.get("verdict") not in VERDICTS:
            raise ValueError(f"Invalid Codex verdict: {row.get('verdict')!r}")
        confidence = row.get("confidence")
        if (
            not isinstance(confidence, int)
            or isinstance(confidence, bool)
            or not 1 <= confidence <= 5
        ):
            raise ValueError(f"Invalid Codex confidence: {confidence!r}")
        if not str(row.get("reason", "")).strip():
            raise ValueError("Codex adjudication reason must be non-empty")
    return [
        {
            **{key: value for key, value in row.items() if key != "task_id"},
            "sample_id": str(expected_by_id[str(row["task_id"])]["sample_id"]),
            "judge_name": "codex_adjudicator",
            "model": model,
            "evidence_hash": expected_by_id[str(row["task_id"])].get("evidence_hash"),
            "adjudication_signature": fingerprint({
                "protocol": ADJUDICATION_PROTOCOL_VERSION,
                "model": model,
                "task": blinded_task(expected_by_id[str(row["task_id"])]),
                "evidence_hash": expected_by_id[str(row["task_id"])].get(
                    "evidence_hash"
                ),
            }),
        }
        for row in rows
    ]


def load_resumed_chunk(
    *,
    expected: list[dict[str, Any]],
    response_path: Path,
    resume_path: Path,
    chunk_signature: str,
    model: str,
    start: int,
) -> list[dict[str, Any]] | None:
    """Load a completed chunk only when its signature and payload validate."""
    if not response_path.exists() or not resume_path.exists():
        return None
    try:
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
        if resume.get("chunk_signature") != chunk_signature:
            return None
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        rows = validate_chunk(expected, payload, model=model)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(
            f"Rejecting invalid Codex resume chunk at offset {start}: "
            f"{type(error).__name__}: {error}"
        )
        return None
    print(f"Reusing validated Codex chunk at offset {start}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--execute-codex", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.input)
    plan = {
        "mode": "execute" if args.execute_codex else "dry-run",
        "model": args.model,
        "tasks": len(tasks),
        "chunk_size": args.chunk_size,
        "output": str(args.output),
        "sandbox": "read-only",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute_codex:
        return

    work = args.output.parent / f".{args.output.stem}.adjudication_v2_chunks"
    work.mkdir(parents=True, exist_ok=True)
    schema_path = work / "adjudication.schema.json"
    atomic_write_json(schema_path, output_schema())
    adjudications = []
    for start in range(0, len(tasks), args.chunk_size):
        chunk = tasks[start:start + args.chunk_size]
        chunk_path = work / f"tasks_{start:05d}.json"
        response_path = work / f"response_{start:05d}.json"
        resume_path = work / f"response_{start:05d}.resume.json"
        visible_chunk = [blinded_task(row) for row in chunk]
        atomic_write_json(chunk_path, visible_chunk)
        chunk_signature = fingerprint({
            "protocol": ADJUDICATION_PROTOCOL_VERSION,
            "model": args.model,
            "schema": output_schema(),
            "tasks": visible_chunk,
            "evidence_hashes": [row.get("evidence_hash") for row in chunk],
        })
        prompt = (
            "Independent grounding adjudication task. Read only the JSON tasks at "
            f"{chunk_path.resolve()} and do not inspect other files. For each item, "
            "judge the claim using only client_stats, train_reference_summary and "
            "field_semantics. Use supported, partially_supported, unsupported, or "
            "not_verifiable. Do not use labels, predictions, source models, prior "
            "judge verdicts, stereotypes, or outside facts. Return every task_id "
            "exactly once."
        )
        rows = load_resumed_chunk(
            expected=chunk,
            response_path=response_path,
            resume_path=resume_path,
            chunk_signature=chunk_signature,
            model=args.model,
            start=start,
        )
        if rows is None:
            command = [
                "codex",
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--cd",
                str(REPO_ROOT),
                "--model",
                args.model,
                "--output-schema",
                str(schema_path.resolve()),
                "--output-last-message",
                str(response_path.resolve()),
                prompt,
            ]
            subprocess.run(command, check=True)
            payload = json.loads(response_path.read_text(encoding="utf-8"))
            rows = validate_chunk(chunk, payload, model=args.model)
            atomic_write_json(resume_path, {"chunk_signature": chunk_signature})
        adjudications.extend(rows)
    atomic_write_json(args.output, adjudications)
    print(f"Saved {len(adjudications)} Codex adjudications -> {args.output}")


if __name__ == "__main__":
    main()

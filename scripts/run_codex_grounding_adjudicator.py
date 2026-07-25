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
                        "sample_id",
                        "verdict",
                        "confidence",
                        "reason",
                    ],
                    "properties": {
                        "sample_id": {"type": "string"},
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
    expected_ids = {str(row["sample_id"]) for row in expected}
    observed_ids = [str(row.get("sample_id")) for row in rows]
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise ValueError("Codex adjudication sample IDs are incomplete or duplicated")
    task_by_id = {str(row["sample_id"]): row for row in expected}
    return [
        {
            **row,
            "sample_id": str(row["sample_id"]),
            "judge_name": "codex_adjudicator",
            "model": model,
            "evidence_hash": task_by_id[str(row["sample_id"])].get(
                "evidence_hash"
            ),
            "adjudication_signature": fingerprint({
                "model": model,
                "task": task_by_id[str(row["sample_id"])],
            }),
        }
        for row in rows
    ]


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

    work = args.output.parent / "codex_chunks"
    work.mkdir(parents=True, exist_ok=True)
    schema_path = work / "adjudication.schema.json"
    atomic_write_json(schema_path, output_schema())
    adjudications = []
    for start in range(0, len(tasks), args.chunk_size):
        chunk = tasks[start:start + args.chunk_size]
        chunk_path = work / f"tasks_{start:05d}.json"
        response_path = work / f"response_{start:05d}.json"
        atomic_write_json(chunk_path, chunk)
        prompt = (
            "Grounding adjudication task. Read the JSON tasks at "
            f"{chunk_path.resolve()}. For each item, use only client_stats, "
            "train_reference_summary and field_semantics. Resolve the judge "
            "disagreement using supported, partially_supported, unsupported, "
            "or not_verifiable. Do not use labels, predictions, stereotypes, "
            "or outside facts. Return every sample_id exactly once."
        )
        command = [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
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
        adjudications.extend(
            validate_chunk(chunk, payload, model=args.model)
        )
    atomic_write_json(args.output, adjudications)
    print(f"Saved {len(adjudications)} Codex adjudications -> {args.output}")


if __name__ == "__main__":
    main()

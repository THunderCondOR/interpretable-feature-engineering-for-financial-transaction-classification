"""Append-only run events and derived status snapshots."""
from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

STAGES = ("stats", "prompts", "explanations", "direct_eval", "claims", "embeddings", "clusters", "features", "ml", "stability", "grounding", "fidelity", "reports")


def append_event(path: str | Path, **event: Any) -> dict[str, Any]:
    record = {"timestamp": time.time(), **event}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_events(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return sorted(records, key=lambda row: row.get("timestamp", 0))


def status_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    cells, errors, concurrency = {}, Counter(), []
    for event in events:
        key = "/".join(str(event.get(field, "unknown")) for field in ("dataset", "model"))
        stage = str(event.get("stage", "unknown"))
        cells[(key, stage)] = event
        for error, count in (event.get("error_counts") or event.get("errors") or {}).items():
            errors[error] += int(count)
        if event.get("concurrency") is not None:
            concurrency.append({"timestamp": event.get("timestamp"), "key": key, "concurrency": event["concurrency"], "mode": event.get("mode")})
    return {"cells": {f"{key}|{stage}": value for (key, stage), value in cells.items()}, "errors": dict(errors), "concurrency_history": concurrency, "updated_at": max((row.get("timestamp", 0) for row in events), default=None)}


def atomic_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)

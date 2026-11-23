"""Small helpers for append-only structured run logs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def append_structured_event(
    path: str | Path | None,
    context: dict[str, Any],
    *,
    event: str,
    **payload: Any,
) -> None:
    """Append one JSONL event without logging prompts or credentials."""
    if not path:
        return
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.time(),
        "event": event,
        **context,
        **payload,
    }
    with open(event_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

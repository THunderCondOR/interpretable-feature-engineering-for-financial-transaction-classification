"""Crash-safe atomic-window scheduler with adaptive concurrency."""
from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable

WindowExecutor = Callable[[list[tuple[int, Any]], int], Awaitable[list[tuple[int, dict]]]]
WindowCommitter = Callable[[list[tuple[int, dict]]], None]

PERMANENT_ERRORS = {
    "AuthenticationError", "PermissionDeniedError", "NotFoundError",
    "BadRequestError", "ConfigurationError", "ModelNotFoundError",
}
TRANSIENT_ERRORS = {
    "APITimeoutError", "APIConnectionError", "InternalServerError",
    "TimeoutError", "ConnectionError", "HTTPStatusError",
}


def is_rate_limit(result: dict) -> bool:
    error = f"{result.get('error_type', '')} {result.get('error', '')}".lower()
    return bool(result.get("rate_limited")) or "ratelimit" in error or "too many requests" in error or " 429" in error


class AtomicAdaptiveScheduler:
    def __init__(self, config: dict, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.high = int(config.get("initial_concurrency", config.get("max_concurrent", 64)))
        self.low = int(config.get("fallback_concurrency", config.get("rate_limit_fallback_concurrent", 10)))
        self.recovery_windows = int(config.get("recovery_clean_batches", config.get("rate_limit_recovery_batches", 10)))
        self.cooldown = float(config.get("cooldown_seconds", config.get("rate_limit_cooldown_seconds", 60)))
        self.backoff = float(config.get("retry_backoff", 2))
        self.max_attempts = int(config.get("max_window_attempts", 20))
        self.signature = str(config.get("generation_signature", "unspecified"))
        self.state_dir = Path(config["scheduler_state_dir"]) if config.get("scheduler_state_dir") else None
        self.events_path = Path(config["events_path"]) if config.get("events_path") else None
        self.sleep = sleep
        if min(self.high, self.low, self.recovery_windows, self.max_attempts) < 1:
            raise ValueError("scheduler concurrency/recovery/attempt values must be positive")

    def _atomic_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _event(self, event: str, **payload) -> None:
        if not self.events_path:
            return
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": time.time(), "event": event, "generation_signature": self.signature, **payload}
        with open(self.events_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _pending_path(self) -> Path | None:
        return self.state_dir / "pending_batch.json" if self.state_dir else None

    def _write_pending(self, *, batch_id, keys, start, concurrency, attempt, mode):
        path = self._pending_path()
        if path:
            self._atomic_json(path, {
                "batch_id": batch_id, "ordered_request_keys": keys, "start_offset": start,
                "concurrency": concurrency, "attempt": attempt, "mode": mode,
                "generation_signature": self.signature,
            })

    def _clear_pending(self):
        path = self._pending_path()
        if path and path.exists():
            path.unlink()

    async def run(self, items, execute_window: WindowExecutor, commit_window: WindowCommitter | None = None, *, request_keys=None):
        indexed = list(enumerate(items))
        keys = [str(key) for key in (request_keys or range(len(items)))]
        if len(keys) != len(items):
            raise ValueError("request_keys length must match items")
        results: list[dict | None] = [None] * len(items)
        offset, mode, clean_low = 0, "high", 0
        pending_path = self._pending_path()
        if pending_path and pending_path.exists():
            try:
                pending = json.loads(pending_path.read_text(encoding="utf-8"))
                pending_keys = pending.get("ordered_request_keys") or []
                if pending.get("generation_signature") == self.signature and pending_keys:
                    offset = keys.index(str(pending_keys[0]))
                    mode = str(pending.get("mode", "high"))
                    self._event("pending_batch_recovered", start=offset, ordered_request_keys=pending_keys)
            except (ValueError, json.JSONDecodeError, KeyError):
                self._event("pending_batch_ignored", reason="incompatible_or_malformed")
        attempt_by_offset: Counter[int] = Counter()

        while offset < len(indexed):
            concurrency = self.high if mode == "high" else self.low
            window = indexed[offset: offset + concurrency]
            window_keys = keys[offset: offset + len(window)]
            attempt_by_offset[offset] += 1
            attempt = attempt_by_offset[offset]
            if attempt > self.max_attempts:
                raise RuntimeError(f"Atomic window at offset {offset} exceeded {self.max_attempts} attempts")
            batch_id = f"{self.signature[:12]}:{offset}:{len(window)}"
            self._write_pending(batch_id=batch_id, keys=window_keys, start=offset, concurrency=concurrency, attempt=attempt, mode=mode)
            self._event("window_started", batch_id=batch_id, start=offset, size=len(window), concurrency=concurrency, mode=mode, attempt=attempt)
            try:
                batch_results = await execute_window(window, concurrency)
            except (asyncio.CancelledError, KeyboardInterrupt):
                self._event("window_interrupted", batch_id=batch_id)
                raise
            except Exception as exc:
                self._event("window_rolled_back", batch_id=batch_id, reason=type(exc).__name__)
                await self.sleep(min(self.cooldown, self.backoff * (2 ** min(attempt - 1, 6))))
                continue

            errors = Counter(str(result.get("error_type") or "UnknownAPIError") for _, result in batch_results if result.get("error"))
            if any(str(result.get("error_type")) in PERMANENT_ERRORS for _, result in batch_results):
                self._event("stage_blocked", batch_id=batch_id, errors=dict(errors))
                raise RuntimeError(f"Permanent API failure: {dict(errors)}")
            if any(is_rate_limit(result) for _, result in batch_results):
                mode, clean_low = "fallback", 0
                self._event("window_rolled_back", batch_id=batch_id, reason="rate_limit", errors=dict(errors), retry_in=self.cooldown)
                await self.sleep(self.cooldown)
                continue
            if errors or len(batch_results) != len(window):
                self._event("window_rolled_back", batch_id=batch_id, reason="transient_or_incomplete", errors=dict(errors))
                await self.sleep(min(self.cooldown, self.backoff * (2 ** min(attempt - 1, 6))))
                continue

            ordered_results = sorted(batch_results, key=lambda pair: pair[0])
            if commit_window:
                try:
                    commit_window(ordered_results)
                except Exception as exc:
                    self._event("window_rolled_back", batch_id=batch_id, reason=f"commit_validation:{type(exc).__name__}")
                    await self.sleep(min(self.cooldown, self.backoff * (2 ** min(attempt - 1, 6))))
                    continue
            for index, result in ordered_results:
                results[index] = result
            offset += len(window)
            self._clear_pending()
            self._event("window_committed", batch_id=batch_id, completed=offset, expected=len(items), concurrency=concurrency, mode=mode)
            if mode == "fallback":
                clean_low += 1
                if clean_low >= self.recovery_windows:
                    mode, clean_low = "high", 0
                    self._event("high_probe_enabled", completed=offset, concurrency=self.high)
        return results

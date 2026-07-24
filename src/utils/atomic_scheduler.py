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
def is_permanent_exception(exc: BaseException) -> bool:
    """Recognize failures that cannot be repaired by retrying the same request."""
    name = type(exc).__name__
    if name in PERMANENT_ERRORS:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "invalid api key",
            "invalid token",
            "authentication failed",
            "permission denied",
            "model not found",
            "unknown model",
            "http 401",
            "http 403",
        )
    )


def is_rate_limit(result: dict) -> bool:
    error = f"{result.get('error_type', '')} {result.get('error', '')}".lower()
    return bool(result.get("rate_limited")) or "ratelimit" in error or "too many requests" in error or " 429" in error


def _error_details(
    batch_results: list[tuple[int, dict]],
    request_keys: list[str],
) -> list[dict[str, Any]]:
    """Return bounded, credential-safe API error details for JSONL events."""
    details = []
    for index, result in batch_results:
        if not result.get("error"):
            continue
        reason = str(result.get("error") or "")
        details.append(
            {
                "request_index": index,
                "request_key": request_keys[index],
                "error_type": str(
                    result.get("error_type") or "UnknownAPIError"
                ),
                "reason": reason[:2000],
                "rate_limited": bool(is_rate_limit(result)),
            }
        )
    return details


class AtomicAdaptiveScheduler:
    def __init__(self, config: dict, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.high = int(config.get("initial_concurrency", config.get("max_concurrent", 64)))
        self.low = int(config.get("fallback_concurrency", config.get("rate_limit_fallback_concurrent", 10)))
        self.serial = int(config.get("minimum_concurrency", 1))
        self.low_rate_limit_attempts = int(config.get("fallback_rate_limit_attempts", 3))
        self.recovery_windows = int(config.get("recovery_clean_batches", config.get("rate_limit_recovery_batches", 10)))
        self.cooldown = float(config.get("cooldown_seconds", config.get("rate_limit_cooldown_seconds", 60)))
        self.backoff = float(config.get("retry_backoff", 2))
        configured_attempts = int(config.get("max_window_attempts", 20))
        # Paid overnight runs explicitly guarded by --until-complete keep
        # retrying repairable windows. Permanent auth/model/config errors are
        # still detected and block immediately below.
        self.max_attempts = None if config.get("until_complete") else configured_attempts
        self.signature = str(config.get("generation_signature", "unspecified"))
        self.event_context = dict(config.get("event_context", {}))
        self.state_dir = Path(config["scheduler_state_dir"]) if config.get("scheduler_state_dir") else None
        self.events_path = Path(config["events_path"]) if config.get("events_path") else None
        self.sleep = sleep
        positive_values = [
            self.high,
            self.low,
            self.serial,
            self.low_rate_limit_attempts,
            self.recovery_windows,
        ]
        if self.max_attempts is not None:
            positive_values.append(self.max_attempts)
        if min(positive_values) < 1:
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
        record = {"timestamp": time.time(), "event": event, "generation_signature": self.signature, **self.event_context, **payload}
        with open(self.events_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _pending_path(self) -> Path | None:
        return self.state_dir / "pending_batch.json" if self.state_dir else None

    def _adaptive_state_path(self) -> Path | None:
        return self.state_dir / "adaptive_state.json" if self.state_dir else None

    def _state_payload(
        self,
        *,
        mode: str,
        clean_windows: int,
        consecutive_low_429: int,
        probe_from: str | None,
    ) -> dict:
        return {
            "mode": mode,
            "clean_windows": clean_windows,
            "consecutive_low_429": consecutive_low_429,
            "probe_from": probe_from,
            "generation_signature": self.signature,
        }

    def _persist_state(self, **state) -> None:
        path = self._adaptive_state_path()
        if path:
            self._atomic_json(path, self._state_payload(**state))

    def _load_state(self) -> dict:
        default = {
            "mode": "high",
            "clean_windows": 0,
            "consecutive_low_429": 0,
            "probe_from": None,
        }
        path = self._adaptive_state_path()
        if not path or not path.is_file():
            return default
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._event("adaptive_state_ignored", reason="malformed")
            return default
        if payload.get("generation_signature") != self.signature:
            self._event("adaptive_state_ignored", reason="signature_changed")
            return default
        mode = str(payload.get("mode", "high"))
        if mode not in {"high", "fallback", "serial"}:
            self._event("adaptive_state_ignored", reason="unknown_mode")
            return default
        return {
            "mode": mode,
            "clean_windows": max(int(payload.get("clean_windows", 0)), 0),
            "consecutive_low_429": max(
                int(payload.get("consecutive_low_429", 0)), 0
            ),
            "probe_from": payload.get("probe_from"),
        }

    def _write_pending(
        self,
        *,
        batch_id,
        keys,
        start,
        concurrency,
        attempt,
        mode,
        clean_windows,
        consecutive_low_429,
        probe_from,
    ):
        path = self._pending_path()
        if path:
            self._atomic_json(path, {
                "batch_id": batch_id, "ordered_request_keys": keys, "start_offset": start,
                "concurrency": concurrency, "attempt": attempt, "mode": mode,
                "clean_windows": clean_windows,
                "consecutive_low_429": consecutive_low_429,
                "probe_from": probe_from,
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
        offset = 0
        state = self._load_state()
        mode = state["mode"]
        clean_windows = state["clean_windows"]
        consecutive_low_429 = state["consecutive_low_429"]
        probe_from = state["probe_from"]
        pending_path = self._pending_path()
        if pending_path and pending_path.exists():
            try:
                pending = json.loads(pending_path.read_text(encoding="utf-8"))
                pending_keys = pending.get("ordered_request_keys") or []
                if pending.get("generation_signature") == self.signature and pending_keys:
                    candidate_offset = keys.index(str(pending_keys[0]))
                    expected_slice = keys[candidate_offset:candidate_offset + len(pending_keys)]
                    if expected_slice == [str(key) for key in pending_keys]:
                        offset = candidate_offset
                        mode = str(pending.get("mode", "high"))
                        clean_windows = max(
                            int(pending.get("clean_windows", clean_windows)), 0
                        )
                        consecutive_low_429 = max(
                            int(
                                pending.get(
                                    "consecutive_low_429",
                                    consecutive_low_429,
                                )
                            ),
                            0,
                        )
                        probe_from = pending.get("probe_from", probe_from)
                        self._event("pending_batch_recovered", start=offset, ordered_request_keys=pending_keys)
                    else:
                        self._event("pending_batch_ignored", reason="request_order_changed")
            except (ValueError, json.JSONDecodeError, KeyError):
                self._event("pending_batch_ignored", reason="incompatible_or_malformed")
        attempt_by_offset: Counter[int] = Counter()

        while offset < len(indexed):
            concurrency = {
                "high": self.high,
                "fallback": self.low,
                "serial": self.serial,
            }[mode]
            window = indexed[offset: offset + concurrency]
            window_keys = keys[offset: offset + len(window)]
            attempt_by_offset[offset] += 1
            attempt = attempt_by_offset[offset]
            if self.max_attempts is not None and attempt > self.max_attempts:
                raise RuntimeError(f"Atomic window at offset {offset} exceeded {self.max_attempts} attempts")
            batch_id = f"{self.signature[:12]}:{offset}:{len(window)}"
            self._write_pending(
                batch_id=batch_id,
                keys=window_keys,
                start=offset,
                concurrency=concurrency,
                attempt=attempt,
                mode=mode,
                clean_windows=clean_windows,
                consecutive_low_429=consecutive_low_429,
                probe_from=probe_from,
            )
            self._event("window_started", batch_id=batch_id, start=offset, size=len(window), concurrency=concurrency, mode=mode, attempt=attempt)
            try:
                batch_results = await execute_window(window, concurrency)
            except (asyncio.CancelledError, KeyboardInterrupt):
                self._event("window_interrupted", batch_id=batch_id)
                raise
            except Exception as exc:
                if is_permanent_exception(exc):
                    self._event(
                        "stage_blocked",
                        batch_id=batch_id,
                        errors={type(exc).__name__: 1},
                    )
                    raise RuntimeError(
                        f"Permanent API failure: {type(exc).__name__}: {exc}"
                    ) from exc
                self._event("window_rolled_back", batch_id=batch_id, reason=type(exc).__name__)
                await self.sleep(min(self.cooldown, self.backoff * (2 ** min(attempt - 1, 6))))
                continue

            errors = Counter(str(result.get("error_type") or "UnknownAPIError") for _, result in batch_results if result.get("error"))
            error_details = _error_details(batch_results, keys)
            if any(str(result.get("error_type")) in PERMANENT_ERRORS for _, result in batch_results):
                self._event(
                    "stage_blocked",
                    batch_id=batch_id,
                    errors=dict(errors),
                    error_details=error_details,
                )
                raise RuntimeError(f"Permanent API failure: {dict(errors)}")
            if any(is_rate_limit(result) for _, result in batch_results):
                previous_mode = mode
                clean_windows = 0
                if mode == "high":
                    mode = "fallback"
                    consecutive_low_429 = 0
                    probe_from = None
                elif mode == "fallback":
                    if probe_from == "serial":
                        mode = "serial"
                        consecutive_low_429 = 0
                        probe_from = None
                    else:
                        consecutive_low_429 += 1
                        if consecutive_low_429 >= self.low_rate_limit_attempts:
                            mode = "serial"
                            consecutive_low_429 = 0
                            probe_from = None
                else:
                    mode = "serial"
                    consecutive_low_429 = 0
                    probe_from = None
                self._persist_state(
                    mode=mode,
                    clean_windows=clean_windows,
                    consecutive_low_429=consecutive_low_429,
                    probe_from=probe_from,
                )
                # pending_batch.json is the crash boundary for the current
                # uncommitted window.  Rewrite it with the *new* limiter state
                # before cooldown so a SIGTERM during sleep cannot resurrect
                # the concurrency level that just failed.
                next_concurrency = {
                    "high": self.high,
                    "fallback": self.low,
                    "serial": self.serial,
                }[mode]
                self._write_pending(
                    batch_id=batch_id,
                    keys=window_keys,
                    start=offset,
                    concurrency=next_concurrency,
                    attempt=attempt,
                    mode=mode,
                    clean_windows=clean_windows,
                    consecutive_low_429=consecutive_low_429,
                    probe_from=probe_from,
                )
                self._event(
                    "window_rolled_back",
                    batch_id=batch_id,
                    reason="rate_limit",
                    errors=dict(errors),
                    error_details=error_details,
                    retry_in=self.cooldown,
                    previous_mode=previous_mode,
                    next_mode=mode,
                    consecutive_low_429=consecutive_low_429,
                )
                await self.sleep(self.cooldown)
                continue
            expected_indices = [index for index, _ in window]
            returned_indices = [index for index, _ in batch_results]
            invalid_indices = (
                len(returned_indices) != len(expected_indices)
                or len(set(returned_indices)) != len(returned_indices)
                or set(returned_indices) != set(expected_indices)
            )
            if errors or invalid_indices:
                self._event(
                    "window_rolled_back",
                    batch_id=batch_id,
                    reason="transient_or_incomplete",
                    errors=dict(errors),
                    error_details=error_details,
                    expected_indices=expected_indices if invalid_indices else None,
                    returned_indices=returned_indices if invalid_indices else None,
                )
                await self.sleep(min(self.cooldown, self.backoff * (2 ** min(attempt - 1, 6))))
                continue

            ordered_results = sorted(batch_results, key=lambda pair: pair[0])
            if commit_window:
                try:
                    commit_window(ordered_results)
                except Exception as exc:
                    self._event(
                        "window_rolled_back",
                        batch_id=batch_id,
                        reason=f"commit_validation:{type(exc).__name__}",
                        exception_type=type(exc).__name__,
                        exception_message=str(exc)[:2000],
                    )
                    await self.sleep(min(self.cooldown, self.backoff * (2 ** min(attempt - 1, 6))))
                    continue
            for index, result in ordered_results:
                results[index] = result
            offset += len(window)
            self._clear_pending()
            committed_mode = mode
            if mode == "serial":
                clean_windows += 1
                if clean_windows >= self.recovery_windows:
                    mode = "fallback"
                    clean_windows = 0
                    probe_from = "serial"
                    self._event(
                        "fallback_probe_enabled",
                        completed=offset,
                        concurrency=self.low,
                    )
            elif mode == "fallback":
                if probe_from == "serial":
                    probe_from = None
                clean_windows += 1
                consecutive_low_429 = 0
                if clean_windows >= self.recovery_windows:
                    mode = "high"
                    clean_windows = 0
                    probe_from = "fallback"
                    self._event(
                        "high_probe_enabled",
                        completed=offset,
                        concurrency=self.high,
                    )
            else:
                clean_windows = 0
                consecutive_low_429 = 0
                probe_from = None
            self._persist_state(
                mode=mode,
                clean_windows=clean_windows,
                consecutive_low_429=consecutive_low_429,
                probe_from=probe_from,
            )
            self._event(
                "window_committed",
                batch_id=batch_id,
                completed=offset,
                expected=len(items),
                concurrency=concurrency,
                mode=committed_mode,
                next_mode=mode,
                clean_windows=clean_windows,
            )
        return results

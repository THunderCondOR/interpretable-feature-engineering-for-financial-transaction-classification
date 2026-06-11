"""
src/utils/async_api.py

Async wrapper for OpenAI-compatible inference APIs.
Reads base_url and api_key from config, expanding environment variables.

Optional llm config keys:
    max_concurrent: 2                  # maximum in-flight requests
    batch_size: 4                      # number of requests scheduled per batch
    request_cooldown_seconds: 0.0      # minimum delay between request starts
    batch_cooldown_seconds: 0.0        # sleep after each batch
    rate_limit_cooldown_seconds: 60.0  # sleep after 429/rate-limit errors
    retry_backoff: 2.0                 # exponential retry backoff base
    max_retries: 3                     # attempts per request
    log_errors: true
    log_retries: true
    raise_on_error: false
    error_preview_chars: 1200
    prompt_preview_chars: 0
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from collections import Counter
from typing import Any

import httpx
import openai
from tqdm import tqdm


class AsyncRateLimiter:
    """Serialize request starts with an optional minimum interval."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.0, float(interval_seconds))
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def wait(self) -> None:
        if self.interval_seconds <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            if delay > 0:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self._next_start = now + self.interval_seconds


def _make_client(llm_config: dict) -> openai.AsyncOpenAI:
    """Build AsyncOpenAI client from config, expanding env vars."""
    base_url = os.path.expandvars(str(llm_config["api_base_url"]))
    api_key = os.path.expandvars(str(llm_config["api_key"]))
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.AsyncClient(verify=False),
    )


def _short(text: Any, limit: int = 1200) -> str:
    """Compact one-line representation for console logs."""
    if text is None:
        return ""
    s = str(text).replace("\n", "\\n")
    if len(s) > limit:
        return s[:limit] + "...<truncated>"
    return s


def _format_exception(exc: BaseException, limit: int = 1200) -> str:
    """Extract useful details from OpenAI/httpx exceptions."""
    parts = [f"{type(exc).__name__}: {_short(exc, limit)}"]

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        parts.append(f"status_code={status_code}")

    request_id = getattr(exc, "request_id", None)
    if request_id:
        parts.append(f"request_id={request_id}")

    body = getattr(exc, "body", None)
    if body:
        parts.append(f"body={_short(body, limit)}")

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            parts.append(f"response_text={_short(response.text, limit)}")
        except Exception:
            pass

    return " | ".join(parts)


def _prompt_preview(messages: list[dict], limit: int) -> str:
    """Return a small user-message preview. Disabled when limit <= 0."""
    if limit <= 0:
        return ""
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_messages:
        return ""
    return _short(user_messages[-1], limit)


def _log(message: str) -> None:
    """Print without breaking tqdm too badly."""
    tqdm.write(message, file=sys.stderr)
    sys.stderr.flush()


def _retry_delay(llm_config: dict, attempt: int, *, rate_limited: bool) -> float:
    backoff = float(llm_config.get("retry_backoff", 2.0)) * (2 ** attempt)
    jitter = random.uniform(0.0, min(3.0, backoff * 0.25))
    if rate_limited:
        cooldown = float(llm_config.get("rate_limit_cooldown_seconds", 60.0))
        return max(cooldown, backoff) + jitter
    return backoff + jitter


async def _query_with_retry(
    messages: list[dict],
    model: str,
    llm_config: dict,
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    rate_limiter: AsyncRateLimiter,
    request_id: int,
    max_retries: int = 3,
) -> dict:
    """Run one chat completion with semaphore, request pacing, and retries."""
    last_exc: BaseException | None = None
    log_errors = bool(llm_config.get("log_errors", True))
    log_retries = bool(llm_config.get("log_retries", True))
    raise_on_error = bool(llm_config.get("raise_on_error", False))
    error_preview_chars = int(llm_config.get("error_preview_chars", 1200))
    prompt_preview_chars = int(llm_config.get("prompt_preview_chars", 0))

    for attempt in range(max_retries):
        try:
            await rate_limiter.wait()
            async with sem:
                t0 = time.monotonic()
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=llm_config.get("temperature", 1.0),
                    top_p=llm_config.get("top_p", 0.9),
                    stream=False,
                )
                return {
                    "response": response,
                    "execution_time": time.monotonic() - t0,
                    "error": None,
                }

        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as exc:
            last_exc = exc
            is_rate_limit = isinstance(exc, openai.RateLimitError)
            err = _format_exception(exc, error_preview_chars)
            if attempt + 1 < max_retries:
                delay = _retry_delay(llm_config, attempt, rate_limited=is_rate_limit)
                if log_retries:
                    _log(
                        f"[API RETRY] request={request_id} attempt={attempt + 1}/{max_retries} "
                        f"sleep={delay:.1f}s model={model} error={err}"
                    )
                await asyncio.sleep(delay)
                continue
            if log_retries:
                _log(
                    f"[API RETRY EXHAUSTED] request={request_id} attempt={attempt + 1}/{max_retries} "
                    f"model={model} error={err}"
                )

        except Exception as exc:
            err = _format_exception(exc, error_preview_chars)
            if log_errors:
                msg = f"[API ERROR] request={request_id} model={model} error={err}"
                preview = _prompt_preview(messages, prompt_preview_chars)
                if preview:
                    msg += f"\nprompt_preview={preview}"
                _log(msg)
            if raise_on_error:
                raise
            return {"response": None, "execution_time": 0.0, "error": err}

    err = _format_exception(last_exc, error_preview_chars) if last_exc else "unknown retryable error"
    if log_errors:
        msg = f"[API ERROR AFTER RETRIES] request={request_id} model={model} error={err}"
        preview = _prompt_preview(messages, prompt_preview_chars)
        if preview:
            msg += f"\nprompt_preview={preview}"
        _log(msg)
    if raise_on_error and last_exc is not None:
        raise last_exc
    return {"response": None, "execution_time": 0.0, "error": err}


async def _run_batch(
    indexed_dialogues: list[tuple[int, list[dict]]],
    model: str,
    llm_config: dict,
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    rate_limiter: AsyncRateLimiter,
) -> list[tuple[int, dict]]:
    """Run a batch and return (original_index, result), logging failures as they complete."""

    async def runner(idx: int, dialogue: list[dict]) -> tuple[int, dict]:
        res = await _query_with_retry(
            dialogue,
            model,
            llm_config,
            client,
            sem,
            rate_limiter,
            request_id=idx,
            max_retries=int(llm_config.get("max_retries", 3)),
        )
        return idx, res

    tasks = [asyncio.create_task(runner(idx, dialogue)) for idx, dialogue in indexed_dialogues]
    results: list[tuple[int, dict]] = []

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="  requests"):
        results.append(await task)

    return results


async def batched_query(
    dialogues: list[list[dict]],
    model: str,
    llm_config: dict,
) -> list[dict]:
    """
    Run dialogues in batches, respecting concurrency, request pacing, and cooldowns.

    Args:
        dialogues: list of message lists, each is one API call.
        model: model name string.
        llm_config: config["llm"] dict.

    Returns:
        List of {response, execution_time, error} dicts, same order as input.
    """
    max_concurrent = int(llm_config.get("max_concurrent", 16))
    batch_size = int(llm_config.get("batch_size", 32))
    batch_cooldown = float(llm_config.get("batch_cooldown_seconds", 0.0))
    request_cooldown = float(llm_config.get("request_cooldown_seconds", 0.0))

    if max_concurrent < 1:
        raise ValueError("llm.max_concurrent must be >= 1")
    if batch_size < 1:
        raise ValueError("llm.batch_size must be >= 1")

    sem = asyncio.Semaphore(max_concurrent)
    rate_limiter = AsyncRateLimiter(request_cooldown)

    client = _make_client(llm_config)
    results: list[dict | None] = [None] * len(dialogues)
    error_counter: Counter[str] = Counter()

    _log(
        f"[API CONFIG] requests={len(dialogues)} max_concurrent={max_concurrent} "
        f"batch_size={batch_size} request_cooldown={request_cooldown}s "
        f"batch_cooldown={batch_cooldown}s"
    )

    try:
        indexed = list(enumerate(dialogues))
        starts = list(range(0, len(indexed), batch_size))
        for batch_i, start in enumerate(tqdm(starts, desc="API batches"), start=1):
            batch = indexed[start : start + batch_size]
            batch_results = await _run_batch(batch, model, llm_config, client, sem, rate_limiter)
            for idx, res in batch_results:
                results[idx] = res
                if res.get("error"):
                    error_counter[_short(res["error"], 240)] += 1
            if batch_cooldown > 0 and batch_i < len(starts):
                _log(f"[API COOLDOWN] sleeping {batch_cooldown:.1f}s before next batch")
                await asyncio.sleep(batch_cooldown)
    finally:
        await client.close()

    final_results = [r if r is not None else {"response": None, "execution_time": 0.0, "error": "missing result"} for r in results]

    n_errors = sum(1 for r in final_results if r.get("error"))
    if n_errors:
        _log(f"[API SUMMARY] {n_errors}/{len(final_results)} requests failed. Top errors:")
        for err, n in error_counter.most_common(5):
            _log(f"  {n}x {err}")

    return final_results

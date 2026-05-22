"""
src/utils/async_api.py

Async wrapper for OpenAI-compatible inference APIs.
Reads base_url and api_key from config (or env vars as fallback).

Usage:
    results = asyncio.run(
        batched_query(dialogues, model, config["llm"])
    )
"""

import asyncio
import os
import time

import httpx
import openai
from tqdm.asyncio import tqdm_asyncio, tqdm


def _make_client(llm_config: dict) -> openai.AsyncOpenAI:
    """Build AsyncOpenAI client from config, expanding env vars."""
    base_url = os.path.expandvars(llm_config["api_base_url"])
    api_key  = os.path.expandvars(llm_config["api_key"])
    return openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.AsyncClient(verify=False),
    )


async def _query_with_retry(
    messages: list[dict],
    model: str,
    llm_config: dict,
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    """Run one chat completion with semaphore, retry on transient errors."""
    backoff = 2.0
    last_exc = None

    for attempt in range(max_retries):
        try:
            async with sem:
                t0 = time.monotonic()
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=llm_config.get("temperature", 1.0),
                    top_p=llm_config.get("top_p", 0.9),
                    stream=False,
                )
                return {"response": response, "execution_time": time.monotonic() - t0, "error": None}
        except (openai.RateLimitError, openai.APITimeoutError) as exc:
            last_exc = exc
            await asyncio.sleep(backoff * (2 ** attempt))
        except Exception as exc:
            # Non-retryable errors
            return {"response": None, "execution_time": 0.0, "error": str(exc)}

    return {"response": None, "execution_time": 0.0, "error": str(last_exc)}


async def batched_query(
    dialogues: list[list[dict]],
    model: str,
    llm_config: dict,
) -> list[dict]:
    """
    Run all dialogues concurrently, respecting max_concurrent limit.

    Args:
        dialogues:  list of message lists (each is one API call)
        model:      model name string
        llm_config: config["llm"] dict with api_base_url, api_key,
                    max_concurrent, batch_size

    Returns:
        List of {response, execution_time, error} dicts, same order as input.
        response is None and error is set on failure.
    """
    max_concurrent = llm_config.get("max_concurrent", 16)
    batch_size     = llm_config.get("batch_size", 32) * max_concurrent
    sem = asyncio.Semaphore(max_concurrent)

    # Create ONE client for all requests in this batch
    client = _make_client(llm_config)

    tasks = [
        _query_with_retry(d, model, llm_config, client, sem)
        for d in dialogues
    ]
    results = []

    for i in tqdm(range(0, len(tasks), batch_size), desc="API batches"):
        batch = tasks[i : i + batch_size]
        batch_results = await tqdm_asyncio.gather(*batch, desc="  requests")
        results.extend(batch_results)

    await client.close()
    return results
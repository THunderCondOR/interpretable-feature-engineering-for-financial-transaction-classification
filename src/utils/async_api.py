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
from functools import wraps

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


def _async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = await func(*args, **kwargs)
        return {"response": result, "execution_time": time.monotonic() - start}
    return wrapper


@_async_timer
async def _query(messages: list[dict], model: str, llm_config: dict) -> dict:
    client = _make_client(llm_config)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=llm_config.get("temperature", 1.0),
        top_p=llm_config.get("top_p", 0.9),
        stream=False,
    )
    return response


async def _limited_query(
    messages: list[dict],
    model: str,
    llm_config: dict,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        return await _query(messages, model, llm_config)


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
        List of {response, execution_time} dicts, same order as input.
    """
    max_concurrent = llm_config.get("max_concurrent", 16)
    batch_size     = llm_config.get("batch_size", 32) * max_concurrent
    sem = asyncio.Semaphore(max_concurrent)

    tasks = [_limited_query(d, model, llm_config, sem) for d in dialogues]
    results = []

    for i in tqdm(range(0, len(tasks), batch_size)):
        batch = tasks[i : i + batch_size]
        batch_results = await tqdm_asyncio.gather(*batch)
        results.extend(batch_results)

    return results

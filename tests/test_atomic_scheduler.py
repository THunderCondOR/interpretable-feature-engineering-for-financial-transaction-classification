import asyncio
import json

import pytest

from src.utils.atomic_scheduler import AtomicAdaptiveScheduler


def ok(index):
    return {"response": f"ok-{index}", "error": None, "error_type": None}


def limited():
    return {"response": None, "error": "429 too many requests", "error_type": "RateLimitError", "rate_limited": True}


def config(tmp_path, name="worker"):
    return {
        "initial_concurrency": 64, "fallback_concurrency": 10,
        "recovery_clean_batches": 10, "cooldown_seconds": 60,
        "generation_signature": name,
        "scheduler_state_dir": str(tmp_path / name),
        "events_path": str(tmp_path / f"{name}.jsonl"),
    }


def test_429_in_64_rolls_back_every_result_then_runs_10x10_and_probes_64(tmp_path):
    calls, commits, sleeps = [], [], []

    async def execute(window, concurrency):
        calls.append((window[0][0], len(window), concurrency))
        if len(calls) == 1:
            return [(index, limited() if position == 0 else ok(index)) for position, (index, _) in enumerate(window)]
        return [(index, ok(index)) for index, _ in window]

    async def sleep(seconds):
        sleeps.append(seconds)

    scheduler = AtomicAdaptiveScheduler(config(tmp_path), sleep=sleep)
    asyncio.run(scheduler.run(list(range(164)), execute, lambda rows: commits.append([i for i, _ in rows])))
    assert calls[0] == (0, 64, 64)
    assert [len(commit) for commit in commits] == [10] * 10 + [64]
    assert commits[0][0] == 0
    assert calls[-1] == (100, 64, 64)
    assert sleeps == [60]


def test_429_at_10_resets_clean_recovery_counter(tmp_path):
    calls, low_successes = [], 0

    async def execute(window, concurrency):
        nonlocal low_successes
        calls.append(concurrency)
        if len(calls) == 1:
            return [(window[0][0], limited())]
        if concurrency == 10:
            low_successes += 1
            if low_successes == 4:
                return [(window[0][0], limited())]
        return [(index, ok(index)) for index, _ in window]

    async def sleep(_seconds):
        return None

    scheduler = AtomicAdaptiveScheduler(config(tmp_path), sleep=sleep)
    asyncio.run(scheduler.run(list(range(140)), execute))
    # Three clean low windows before the second 429 do not count toward recovery.
    assert calls[:5] == [64, 10, 10, 10, 10]
    assert calls[5:15] == [10] * 10
    assert calls[15] == 64


def test_crash_keeps_pending_and_restart_retries_first_key(tmp_path):
    state = config(tmp_path)

    async def crash(_window, _concurrency):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(AtomicAdaptiveScheduler(state).run(list(range(5)), crash, request_keys=[f"k{i}" for i in range(5)]))
    pending = tmp_path / "worker" / "pending_batch.json"
    assert json.loads(pending.read_text())["ordered_request_keys"][0] == "k0"
    seen = []

    async def recover(window, concurrency):
        seen.append((window[0][0], concurrency))
        return [(index, ok(index)) for index, _ in window]

    asyncio.run(AtomicAdaptiveScheduler(state).run(list(range(5)), recover, request_keys=[f"k{i}" for i in range(5)]))
    assert seen == [(0, 64)]
    assert not pending.exists()


def test_model_scheduler_states_are_independent(tmp_path):
    qwen = AtomicAdaptiveScheduler(config(tmp_path, "qwen"), sleep=lambda _: asyncio.sleep(0))
    gpt = AtomicAdaptiveScheduler(config(tmp_path, "gpt"), sleep=lambda _: asyncio.sleep(0))
    qwen_calls, gpt_calls = [], []

    async def qwen_execute(window, concurrency):
        qwen_calls.append(concurrency)
        if len(qwen_calls) == 1:
            return [(window[0][0], limited())]
        return [(index, ok(index)) for index, _ in window]

    async def gpt_execute(window, concurrency):
        gpt_calls.append(concurrency)
        return [(index, ok(index)) for index, _ in window]

    asyncio.run(qwen.run(list(range(20)), qwen_execute))
    asyncio.run(gpt.run(list(range(20)), gpt_execute))
    assert qwen_calls[:2] == [64, 10]
    assert gpt_calls == [64]


def test_commit_validation_failure_rolls_back_window(tmp_path):
    attempts, commits = 0, 0

    async def execute(window, concurrency):
        nonlocal attempts
        attempts += 1
        return [(index, ok(index)) for index, _ in window]

    def commit(_rows):
        nonlocal commits
        commits += 1
        if commits == 1:
            raise ValueError("parser failed")

    scheduler = AtomicAdaptiveScheduler(config(tmp_path), sleep=lambda _: asyncio.sleep(0))
    asyncio.run(scheduler.run(list(range(8)), execute, commit))
    assert attempts == 2
    assert commits == 2


def test_duplicate_or_wrong_result_indices_roll_back_window(tmp_path):
    attempts = 0

    async def execute(window, _concurrency):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first = window[0][0]
            return [(first, ok(first)) for _ in window]
        return [(index, ok(index)) for index, _ in window]

    scheduler = AtomicAdaptiveScheduler(
        config(tmp_path), sleep=lambda _: asyncio.sleep(0)
    )
    result = asyncio.run(scheduler.run(list(range(6)), execute))
    assert attempts == 2
    assert [row["response"] for row in result] == [f"ok-{index}" for index in range(6)]


def test_pending_batch_with_changed_request_order_is_ignored(tmp_path):
    state = config(tmp_path)
    state_dir = tmp_path / "worker"
    state_dir.mkdir()
    (state_dir / "pending_batch.json").write_text(
        json.dumps(
            {
                "ordered_request_keys": ["k2", "k3"],
                "generation_signature": "worker",
                "mode": "fallback",
            }
        ),
        encoding="utf-8",
    )
    starts = []

    async def execute(window, concurrency):
        starts.append((window[0][0], concurrency))
        return [(index, ok(index)) for index, _ in window]

    scheduler = AtomicAdaptiveScheduler(state, sleep=lambda _: asyncio.sleep(0))
    asyncio.run(
        scheduler.run(
            list(range(4)),
            execute,
            request_keys=["k0", "k2", "k1", "k3"],
        )
    )
    assert starts == [(0, 64)]
    events = [
        json.loads(line)
        for line in (tmp_path / "worker.jsonl").read_text().splitlines()
    ]
    assert any(
        event.get("reason") == "request_order_changed"
        for event in events
        if event["event"] == "pending_batch_ignored"
    )


def test_permanent_api_error_blocks_without_retrying(tmp_path):
    attempts = 0

    async def execute(window, _concurrency):
        nonlocal attempts
        attempts += 1
        return [
            (
                window[0][0],
                {
                    "response": None,
                    "error": "invalid token",
                    "error_type": "AuthenticationError",
                },
            )
        ]

    scheduler = AtomicAdaptiveScheduler(
        config(tmp_path), sleep=lambda _: asyncio.sleep(0)
    )
    with pytest.raises(RuntimeError, match="Permanent API failure"):
        asyncio.run(scheduler.run([1], execute))
    assert attempts == 1

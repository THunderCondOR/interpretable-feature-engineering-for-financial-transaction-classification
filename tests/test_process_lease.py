import json

import pytest

from src.utils.process_lease import LeaseInUseError, ProcessLease
from scripts.run_cv_llm_queue import active_model_queues


def test_process_lease_blocks_second_owner_and_can_be_reacquired(tmp_path):
    path = tmp_path / "gpt_oss.lock"
    first = ProcessLease(path, {"model": "gpt_oss", "run_id": "first"})
    second = ProcessLease(path, {"model": "gpt_oss", "run_id": "second"})

    first.acquire()
    with pytest.raises(LeaseInUseError, match="first"):
        second.acquire()
    assert json.loads(path.read_text())["state"] == "held"

    first.release()
    second.acquire()
    assert json.loads(path.read_text())["run_id"] == "second"
    second.release()


def test_qwen_and_gpt_use_independent_leases(tmp_path):
    qwen = ProcessLease(tmp_path / "qwen.lock", {"model": "qwen"})
    gpt = ProcessLease(tmp_path / "gpt_oss.lock", {"model": "gpt_oss"})
    qwen.acquire()
    gpt.acquire()
    assert json.loads((tmp_path / "qwen.lock").read_text())["state"] == "held"
    assert json.loads((tmp_path / "gpt_oss.lock").read_text())["state"] == "held"
    qwen.release()
    gpt.release()


def test_active_queue_scan_is_model_specific_and_ignores_own_pid(tmp_path):
    proc = tmp_path / "proc"
    for pid, argv in {
        10: ["python", "scripts/run_cv_llm_queue.py", "--model", "gpt_oss"],
        11: ["python", "scripts/run_cv_llm_queue.py", "--model", "qwen"],
        12: ["python", "scripts/run_cv_queue_watchdog.py", "--model", "gpt_oss"],
    }.items():
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        (directory / "cmdline").write_bytes(
            b"\0".join(value.encode() for value in argv) + b"\0"
        )

    assert [row["pid"] for row in active_model_queues(
        "gpt_oss", proc_root=proc, own_pid=99
    )] == [10]
    assert active_model_queues("gpt_oss", proc_root=proc, own_pid=10) == []

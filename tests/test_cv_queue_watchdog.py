import json
import signal

import pytest

from scripts.run_cv_queue_watchdog import (
    WatchdogShutdown,
    consume_events,
    full_connection_failure,
    request_shutdown,
)


def test_only_full_connection_windows_trigger_restart():
    sizes = {"batch": 64}
    timeout = {
        "event": "window_rolled_back",
        "reason": "transient_or_incomplete",
        "batch_id": "batch",
        "errors": {"APITimeoutError": 64},
    }
    assert full_connection_failure(timeout, sizes)
    timeout["errors"]["APITimeoutError"] = 63
    assert not full_connection_failure(timeout, sizes)
    timeout["errors"] = {"ParserError": 64}
    assert not full_connection_failure(timeout, sizes)


def test_event_consumer_uses_current_window_size(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {
            "event": "window_started",
            "batch_id": "a",
            "size": 10,
        },
        {
            "event": "window_rolled_back",
            "reason": "transient_or_incomplete",
            "batch_id": "a",
            "errors": {"APIConnectionError": 10},
        },
    ]
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    offset, restart, event = consume_events(path, 0, {})
    assert offset == path.stat().st_size
    assert restart
    assert event["batch_id"] == "a"


def test_terminal_signal_enters_cleanup_path():
    with pytest.raises(WatchdogShutdown) as caught:
        request_shutdown(signal.SIGTERM, None)
    assert caught.value.signum == signal.SIGTERM

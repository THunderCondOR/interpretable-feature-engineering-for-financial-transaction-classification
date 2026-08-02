"""Small cross-process leases for long-running experiment queues."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import time
from pathlib import Path
from typing import Any


class LeaseInUseError(RuntimeError):
    """Raised when another process already owns an exclusive lease."""


class ProcessLease:
    """Hold an advisory file lock until release or process termination."""

    def __init__(self, path: Path, metadata: dict[str, Any]) -> None:
        self.path = path
        self.metadata = metadata
        self._file = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            file.seek(0)
            holder = file.read().strip() or "unknown holder"
            file.close()
            raise LeaseInUseError(
                f"Queue lease is already held: {self.path}; holder={holder}"
            ) from exc
        payload = {
            **self.metadata,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": time.time(),
            "state": "held",
        }
        file.seek(0)
        file.truncate()
        json.dump(payload, file, ensure_ascii=False, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
        self._file = file

    def release(self) -> None:
        if self._file is None:
            return
        try:
            self._file.seek(0)
            self._file.truncate()
            json.dump(
                {
                    **self.metadata,
                    "pid": os.getpid(),
                    "released_at": time.time(),
                    "state": "released",
                },
                self._file,
                ensure_ascii=False,
                sort_keys=True,
            )
            self._file.flush()
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

"""Shared, provenance-heavy helpers for isolated reviewer benchmarks."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from src.experiments.artifacts import atomic_write_json, fingerprint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(url: str, path: Path, expected_sha256: str) -> Path:
    """Download once and reject both stale local and corrupted remote files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and sha256(path) == expected_sha256:
        return path
    temporary = path.with_suffix(path.suffix + ".download")
    temporary.unlink(missing_ok=True)
    urllib.request.urlretrieve(url, temporary)
    observed = sha256(temporary)
    if observed != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for {url}: {observed} != {expected_sha256}"
        )
    temporary.replace(path)
    return path


def stable_ids(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values}, key=lambda x: (len(x), x))


def write_ids(path: Path, values: Iterable[Any]) -> Path:
    atomic_write_json(path, stable_ids(values))
    return path


def id_hash(values: Iterable[Any]) -> str:
    return fingerprint(stable_ids(values))


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload

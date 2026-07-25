"""Stage-scoped manifests for artifacts derived from immutable API runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.artifacts import (
    atomic_write_json,
    file_sha256,
    files_fingerprint,
    fingerprint,
    git_revision,
)


def source_contract(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claim_paths = {
        split: root / f"claims_{split}.jsonl"
        for split in ("train", "val", "test")
    }
    missing = [str(path) for path in claim_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source claims: {missing}")
    return {
        "source_root": str(root),
        "run_id": manifest.get("run_id"),
        "dataset": manifest.get("config", {}).get("dataset", {}).get("name"),
        "model_slug": manifest.get("config", {}).get("experiment", {}).get(
            "model_slug"
        ),
        "variant": manifest.get("variant"),
        "seed": manifest.get("seed"),
        "source_manifest_sha256": manifest.get("manifest_sha256"),
        "source_manifest_file_sha256": file_sha256(manifest_path),
        "claims": files_fingerprint(claim_paths.values()),
    }


def stage_identity(
    *,
    stage: str,
    source: dict[str, Any],
    inputs: Any,
    configuration: Any,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    identity = {
        "stage_manifest_version": 1,
        "stage": stage,
        "source": source,
        "inputs": inputs,
        "configuration": configuration,
        "git_revision": git_revision(repo_root),
    }
    return {
        **identity,
        "stage_signature": fingerprint(identity),
    }


def compatible_stage(path: str | Path, expected: dict[str, Any]) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    outputs = [Path(item) for item in existing.get("outputs", [])]
    if (
        existing.get("state") != "completed"
        or existing.get("stage_signature") != expected.get("stage_signature")
        or not outputs
        or not all(item.is_file() for item in outputs)
    ):
        return False
    return files_fingerprint(outputs) == existing.get("output_files")


def complete_stage(
    path: str | Path,
    identity: dict[str, Any],
    *,
    outputs: list[str | Path],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = [Path(item) for item in outputs]
    missing = [str(item) for item in paths if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot complete stage with missing outputs: {missing}")
    payload = {
        **identity,
        "state": "completed",
        "outputs": [str(item) for item in paths],
        "output_files": files_fingerprint(paths),
        "metrics": metrics or {},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(path, payload)
    return payload


def verify_source_unchanged(before: dict[str, Any]) -> None:
    after = source_contract(before["source_root"])
    stable_keys = (
        "source_manifest_file_sha256",
        "source_manifest_sha256",
        "claims",
    )
    changed = [key for key in stable_keys if before.get(key) != after.get(key)]
    if changed:
        raise RuntimeError(
            f"Immutable API source changed during derived run: {changed}"
        )

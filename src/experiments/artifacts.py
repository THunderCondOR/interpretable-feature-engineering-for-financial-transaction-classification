"""Content-addressed experiment artifacts.

The original pipeline resumed work by customer ID alone.  That is unsafe when a
prompt, model, source explanation, or feature configuration changes: a record can
look complete while belonging to a different experiment.  This module provides a
small, dependency-free signature layer shared by every pipeline stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


SIGNATURE_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing and manifests."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def files_fingerprint(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Return stable content identities for an ordered collection of files."""
    identities: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            identities[str(path)] = {"exists": False}
            continue
        stat = path.stat()
        identities[str(path)] = {
            "exists": True,
            "size": int(stat.st_size),
            "sha256": file_sha256(path),
        }
    return identities


def stage_signature(stage: str, *, inputs: Any, configuration: Any) -> str:
    return fingerprint(
        {
            "signature_version": SIGNATURE_VERSION,
            "stage": stage,
            "inputs": inputs,
            "configuration": configuration,
        }
    )


def prompt_signature(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    decoding: dict[str, Any],
    sample_id: int = 0,
) -> str:
    return stage_signature(
        "explanation_generation",
        inputs={"system_prompt": system_prompt, "user_prompt": user_prompt},
        configuration={
            "model": model,
            "decoding": decoding,
            "sample_id": int(sample_id),
        },
    )


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=_json_default)
    temporary.replace(path)


def git_revision(repo_root: str | Path = ".") -> str | None:
    """Best-effort Git revision; manifests still work outside Git checkouts."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return None


def runtime_identity() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pid": os.getpid(),
    }


def build_run_manifest(config: dict[str, Any], *, repo_root: str | Path = ".") -> dict[str, Any]:
    split_paths = list(config.get("dataset", {}).get("splits", {}).values())
    prompt_cfg = config.get("prompts", {})
    base_dir = Path(prompt_cfg.get("base_dir", "."))
    prompt_paths = [
        base_dir / prompt_cfg[key]
        for key in ("system", "user", "claims_system", "claims_user")
        if prompt_cfg.get(key)
    ]
    experiment = config.get("experiment", {})
    return {
        "manifest_version": 1,
        "run_id": experiment.get("run_id", config.get("dataset", {}).get("name", "run")),
        "variant": experiment.get("variant", "legacy-compatible"),
        "seed": int(experiment.get("seed", 42)),
        "git_revision": git_revision(repo_root),
        "config": config,
        "config_sha256": fingerprint(config),
        "dataset_files": files_fingerprint(split_paths),
        "prompt_files": files_fingerprint(prompt_paths),
        "runtime": runtime_identity(),
    }


def ensure_run_manifest(config: dict[str, Any], *, repo_root: str | Path = ".") -> Path:
    out_dir = Path(config["output"]["base_dir"])
    path = out_dir / "run_manifest.json"
    candidate = build_run_manifest(config, repo_root=repo_root)
    if path.exists():
        with open(path, encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("config_sha256") != candidate["config_sha256"]:
            raise RuntimeError(
                f"Output directory {out_dir} belongs to an incompatible experiment: "
                f"manifest config hash {existing.get('config_sha256')} != "
                f"{candidate['config_sha256']}. Choose a versioned output directory."
            )
        return path
    atomic_write_json(path, candidate)
    return path

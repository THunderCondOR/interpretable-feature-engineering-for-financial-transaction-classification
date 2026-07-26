"""Content-addressed experiment artifacts.

The original pipeline resumed work by customer ID alone.  That is unsafe when a
prompt, model, source explanation, or feature configuration changes: a record can
look complete while belonging to a different experiment.  This module provides a
small, dependency-free signature layer shared by every pipeline stage.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
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
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
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
    packages = {}
    for name in (
        "numpy", "pandas", "scikit-learn", "scipy", "sentence-transformers",
        "torch", "xgboost", "optuna", "openai", "httpx", "plotly", "jinja2",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def build_run_manifest(config: dict[str, Any], *, repo_root: str | Path = ".") -> dict[str, Any]:
    """Capture immutable inputs and a content identity for safe reuse."""
    split_paths = list(config.get("dataset", {}).get("splits", {}).values())
    prompt_cfg = config.get("prompts", {})
    base_dir = Path(prompt_cfg.get("base_dir", "."))
    prompt_paths = [
        base_dir / prompt_cfg[key]
        for key in ("system", "user", "claims_system", "claims_user")
        if prompt_cfg.get(key)
    ]
    experiment = config.get("experiment", {})
    dataset_files = files_fingerprint(split_paths)
    selection_paths = [
        value
        for value in config.get("dataset", {}).get("client_ids_by_split", {}).values()
        if isinstance(value, (str, Path))
    ]
    client_selection_files = files_fingerprint(selection_paths)
    prompt_files = files_fingerprint(prompt_paths)
    prompt_context = {
        "split": config.get("pipeline", {}).get(
            "prompt_context_split", "train"
        ),
        "population": config.get("pipeline", {}).get(
            "prompt_context_population", "full_train_split"
        ),
    }
    revision = git_revision(repo_root)
    runtime = runtime_identity()
    config_sha256 = fingerprint(config)
    identity_payload = {
        "manifest_version": 2,
        "config_sha256": config_sha256,
        "dataset_files": dataset_files,
        "client_selection_files": client_selection_files,
        "prompt_files": prompt_files,
        "prompt_context": prompt_context,
        "git_revision": revision,
        "packages": runtime["packages"],
    }
    return {
        "manifest_version": 2,
        "run_id": experiment.get("run_id", config.get("dataset", {}).get("name", "run")),
        "variant": experiment.get("variant", "legacy-compatible"),
        "seed": int(experiment.get("seed", 42)),
        "model_id": config.get("generation", {}).get(
            "model", config.get("llm", {}).get("default_model")
        ),
        "git_revision": revision,
        "config": config,
        "config_sha256": config_sha256,
        "dataset_files": dataset_files,
        "client_selection_files": client_selection_files,
        "prompt_files": prompt_files,
        "prompt_context": prompt_context,
        "few_shot_configuration": config.get("pipeline", {}),
        "runtime": runtime,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": fingerprint(identity_payload),
    }


def ensure_run_manifest(config: dict[str, Any], *, repo_root: str | Path = ".") -> Path:
    out_dir = Path(config["output"]["base_dir"])
    path = out_dir / "manifest.json"
    candidate = build_run_manifest(config, repo_root=repo_root)
    if path.exists():
        with open(path, encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("manifest_sha256") != candidate["manifest_sha256"]:
            # A scheduler-only hotfix may need to resume an already paid,
            # content-addressed generation without rewriting its immutable
            # provenance. This opt-in is intentionally environment-scoped and
            # accepts only a Git-revision difference; configs, inputs, prompts,
            # client selections, and package versions must remain identical.
            resume_across_revision = os.environ.get(
                "ALLOW_SCHEDULER_CODE_RESUME", ""
            ).lower() in {"1", "true", "yes"}
            stable_fields = (
                "config_sha256",
                "dataset_files",
                "client_selection_files",
                "prompt_files",
                "prompt_context",
            )
            same_generation_contract = all(
                existing.get(field) == candidate.get(field)
                for field in stable_fields
            ) and existing.get("runtime", {}).get("packages") == candidate.get(
                "runtime", {}
            ).get("packages")
            if resume_across_revision and same_generation_contract:
                return path
            raise RuntimeError(
                f"Output directory {out_dir} belongs to an incompatible experiment: "
                f"manifest identity {existing.get('manifest_sha256')} != "
                f"{candidate['manifest_sha256']}. Dataset, prompt, code, package, or "
                "configuration content changed; choose a new versioned output directory."
            )
        return path
    atomic_write_json(path, candidate)
    return path

#!/usr/bin/env python3
"""Safely migrate a run manifest after a runtime-only code fix.

The migration is deliberately narrow: dataset, config, prompts, packages and
all saved explanation signatures must remain identical.  Dry-run is the
default.  This must not be used for prompt, model or decoding changes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.experiments.artifacts import (
    atomic_write_json,
    build_run_manifest,
    prompt_signature,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STABLE_MANIFEST_FIELDS = (
    "manifest_version",
    "run_id",
    "variant",
    "seed",
    "model_id",
    "config_sha256",
    "dataset_files",
    "client_selection_files",
    "prompt_files",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_explanations(config: dict[str, Any]) -> dict[str, int]:
    output = config["output"]
    paths = output.get("paths_by_split", {})
    llm = {
        **config.get("llm", {}),
        **config.get("execution", {}),
        **config.get("generation", {}),
    }
    model = str(llm.get("model", llm["default_model"]))
    decoding = {
        "temperature": llm.get("temperature", 1.0),
        "top_p": llm.get("top_p", 0.9),
        "max_tokens": llm.get("max_tokens", 2048),
        "seed": llm.get("seed"),
        "extra_body": llm.get("extra_body"),
    }
    validated: dict[str, int] = {}
    for split, split_paths in paths.items():
        prompt_path = Path(split_paths["prompts"])
        explanation_path = Path(split_paths["explanations"])
        prompts = {
            int(record["customer_id"]): record
            for record in _load_jsonl(prompt_path)
        }
        explanations = _load_jsonl(explanation_path)
        seen: set[tuple[int, int]] = set()
        for record in explanations:
            customer_id = int(record["customer_id"])
            sample_id = int(record.get("sample_id", 0))
            key = (customer_id, sample_id)
            if key in seen:
                raise RuntimeError(
                    f"Duplicate explanation key {key} in {explanation_path}"
                )
            seen.add(key)
            prompt = prompts.get(customer_id)
            if prompt is None:
                raise RuntimeError(
                    f"Missing prompt for client {customer_id} in {prompt_path}"
                )
            expected = prompt_signature(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
                model=model,
                decoding=decoding,
                sample_id=sample_id,
            )
            if record.get("generation_signature") != expected:
                raise RuntimeError(
                    f"Incompatible generation signature for {key}"
                )
            if record.get("prompt_hash") != prompt.get("prompt_hash"):
                raise RuntimeError(f"Prompt hash mismatch for {key}")
            if record.get("error") or not str(
                record.get("explanation", "")
            ).strip():
                raise RuntimeError(f"Saved explanation {key} is not successful")
        claims_path = Path(split_paths["claims"])
        if claims_path.is_file() and claims_path.stat().st_size:
            raise RuntimeError(
                f"Refusing runtime-only migration with existing claims: "
                f"{claims_path}"
            )
        validated[split] = len(explanations)
    return validated


def migrate(config_path: Path, *, reason: str, execute: bool) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["output"]["base_dir"]) / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = build_run_manifest(config, repo_root=REPO_ROOT)

    changed = [
        field
        for field in STABLE_MANIFEST_FIELDS
        if existing.get(field) != candidate.get(field)
    ]
    if changed:
        raise RuntimeError(
            f"Not a runtime-only migration; changed fields: {changed}"
        )
    if existing.get("runtime", {}).get("packages") != candidate.get(
        "runtime", {}
    ).get("packages"):
        raise RuntimeError("Package versions changed")

    validated = _validate_explanations(config)
    migration = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "from_manifest_sha256": existing.get("manifest_sha256"),
        "to_manifest_sha256": candidate["manifest_sha256"],
        "from_git_revision": existing.get("git_revision"),
        "to_git_revision": candidate.get("git_revision"),
        "validated_explanations": validated,
    }
    candidate["runtime_migrations"] = [
        *existing.get("runtime_migrations", []),
        migration,
    ]
    if execute:
        atomic_write_json(manifest_path, candidate)
    return {
        "mode": "execute" if execute else "dry-run",
        "config": str(config_path),
        "manifest": str(manifest_path),
        "migration": migration,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    results = [
        migrate(Path(path), reason=args.reason, execute=args.execute)
        for path in args.config
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

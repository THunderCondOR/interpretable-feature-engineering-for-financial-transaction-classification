#!/usr/bin/env python3
"""Fail-fast validation for the full overnight LLM queues.

The command is deliberately read-only.  It validates the interpreter, API
environment, model profiles, dataset inputs and Git provenance before the
launcher creates logs or tmux sessions.  Network calls are made only when the
explicit ``--probe`` flag is present.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


EXPECTED_CLIENT_COUNTS: dict[str, dict[str, int]] = {
    "gender": {"train": 6_720, "val": 840, "test": 840},
    "age": {"train": 24_000, "val": 3_000, "test": 3_000},
    "rosbank": {"train": 4_000, "val": 500, "test": 500},
}
REQUIRED_IMPORTS = ("pandas", "sklearn", "xgboost", "optuna", "openai", "yaml")
UNRESOLVED_ENV = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*")


class PreflightError(RuntimeError):
    """A launch prerequisite is missing or incompatible."""


def _load_yaml(path: Path) -> dict[str, Any]:
    yaml = importlib.import_module("yaml")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise PreflightError(f"YAML root must be a mapping: {path}")
    return payload


def validate_imports(modules: tuple[str, ...] = REQUIRED_IMPORTS) -> None:
    missing: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except (ImportError, ModuleNotFoundError):
            missing.append(module)
    if missing:
        raise PreflightError(
            f"Python {sys.executable} is missing required packages: {', '.join(missing)}"
        )


def _required_environment(environ: dict[str, str] | os._Environ[str]) -> tuple[str, str]:
    values: list[str] = []
    for name in ("API_BASE_URL", "API_KEY"):
        value = str(environ.get(name, "")).strip()
        if not value:
            raise PreflightError(f"Required environment variable is empty: {name}")
        if UNRESOLVED_ENV.search(value):
            raise PreflightError(f"Required environment variable is unresolved: {name}")
        values.append(value)
    base_url, api_key = values
    if not base_url.startswith(("http://", "https://")):
        raise PreflightError("API_BASE_URL must be an http(s) URL")
    return base_url, api_key


def validate_git_clean(repo_root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        preview = "\n".join(status.splitlines()[:12])
        raise PreflightError(
            "Refusing paid API execution from a dirty worktree. Commit or remove these changes:\n"
            + preview
        )
    return revision


def validate_model_profiles(paths: list[Path]) -> list[dict[str, str]]:
    profiles: list[dict[str, str]] = []
    seen_slugs: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise PreflightError(f"Model profile does not exist: {path}")
        profile = _load_yaml(path)
        try:
            slug = str(profile["experiment"]["model_slug"]).strip()
            generation_model = str(profile["generation"]["model"]).strip()
            claims_model = str(profile["claims_generation"]["model"]).strip()
        except (KeyError, TypeError) as exc:
            raise PreflightError(f"Incomplete model profile: {path}: {exc}") from exc
        if not slug or not generation_model or not claims_model:
            raise PreflightError(f"Empty model identifier in profile: {path}")
        if slug in seen_slugs:
            raise PreflightError(f"Duplicate model_slug across profiles: {slug}")
        seen_slugs.add(slug)
        profiles.append(
            {
                "slug": slug,
                "generation_model": generation_model,
                "claims_model": claims_model,
                "path": str(path),
            }
        )
    if seen_slugs != {"qwen", "gpt_oss"}:
        raise PreflightError(
            f"Expected qwen and gpt_oss profiles, found: {sorted(seen_slugs)}"
        )
    return profiles


def _resolve_repo_path(repo_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def validate_datasets(
    repo_root: Path,
    config_paths: list[Path],
    expected_counts: dict[str, dict[str, int]] = EXPECTED_CLIENT_COUNTS,
) -> dict[str, dict[str, int]]:
    pandas = importlib.import_module("pandas")
    observed: dict[str, dict[str, int]] = {}
    all_ids: dict[str, dict[str, set[Any]]] = {}
    seen_datasets: set[str] = set()

    for config_path in config_paths:
        if not config_path.is_file():
            raise PreflightError(f"Dataset config does not exist: {config_path}")
        config = _load_yaml(config_path)
        try:
            dataset = str(config["dataset"]["name"])
            id_column = str(config["dataset"]["columns"]["customer_id"])
            splits = config["dataset"]["splits"]
        except (KeyError, TypeError) as exc:
            raise PreflightError(f"Incomplete dataset config: {config_path}: {exc}") from exc
        if dataset in seen_datasets:
            raise PreflightError(f"Duplicate dataset config: {dataset}")
        seen_datasets.add(dataset)
        if dataset not in expected_counts:
            raise PreflightError(f"Unexpected dataset in overnight queue: {dataset}")
        if set(splits) != {"train", "val", "test"}:
            raise PreflightError(f"Dataset {dataset} must define train/val/test splits")

        prompt_config = config.get("prompts", {})
        prompt_base = _resolve_repo_path(repo_root, str(prompt_config.get("base_dir", ".")))
        for key in ("system", "user", "claims_system", "claims_user"):
            value = prompt_config.get(key)
            prompt_path = prompt_base / str(value) if value else None
            if prompt_path is None or not prompt_path.is_file():
                raise PreflightError(f"Missing {dataset} prompt {key}: {prompt_path}")

        observed[dataset] = {}
        all_ids[dataset] = {}
        for split in ("train", "val", "test"):
            split_path = _resolve_repo_path(repo_root, str(splits[split]))
            if not split_path.is_file():
                raise PreflightError(f"Missing {dataset}/{split} split: {split_path}")
            frame = pandas.read_csv(split_path, usecols=[id_column])
            if frame[id_column].isna().any():
                raise PreflightError(f"Null customer IDs in {dataset}/{split}")
            identifiers = set(frame[id_column].tolist())
            count = len(identifiers)
            expected = expected_counts[dataset][split]
            if count != expected:
                raise PreflightError(
                    f"Unexpected unique client count for {dataset}/{split}: "
                    f"observed={count}, expected={expected}"
                )
            observed[dataset][split] = count
            all_ids[dataset][split] = identifiers

        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = all_ids[dataset][left] & all_ids[dataset][right]
            if overlap:
                raise PreflightError(
                    f"Client leakage between {dataset}/{left} and {dataset}/{right}: "
                    f"{len(overlap)} IDs"
                )

    if seen_datasets != set(expected_counts):
        raise PreflightError(
            f"Expected dataset configs {sorted(expected_counts)}, found {sorted(seen_datasets)}"
        )
    return observed


def probe_models(
    profiles: list[dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    client_factory: Callable[..., Any] | None = None,
) -> list[str]:
    """Issue one minimal request per generation model; never expose credentials."""
    if client_factory is None:
        openai = importlib.import_module("openai")
        client_factory = openai.OpenAI
    client = client_factory(base_url=base_url, api_key=api_key)
    completed: list[str] = []
    try:
        for profile in profiles:
            client.chat.completions.create(
                model=profile["generation_model"],
                messages=[{"role": "user", "content": "Reply with OK."}],
                max_tokens=8,
            )
            completed.append(profile["slug"])
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return completed


def run_preflight(
    *,
    repo_root: Path,
    qwen_config: Path,
    gpt_config: Path,
    dataset_configs: list[Path],
    environ: dict[str, str] | os._Environ[str],
    expected_counts: dict[str, dict[str, int]] = EXPECTED_CLIENT_COUNTS,
    check_git: bool = True,
    probe: bool = False,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    validate_imports()
    base_url, api_key = _required_environment(environ)
    profiles = validate_model_profiles([qwen_config, gpt_config])
    counts = validate_datasets(repo_root, dataset_configs, expected_counts)
    revision = validate_git_clean(repo_root) if check_git else "test-no-git-check"
    probed = (
        probe_models(
            profiles,
            base_url=base_url,
            api_key=api_key,
            client_factory=client_factory,
        )
        if probe
        else []
    )
    return {
        "status": "ok",
        "git_revision": revision,
        "models": [profile["slug"] for profile in profiles],
        "datasets": counts,
        "clients_per_model": sum(sum(splits.values()) for splits in counts.values()),
        "estimated_api_requests": 174_800,
        "probed_models": probed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--qwen-config", type=Path, required=True)
    parser.add_argument("--gpt-config", type=Path, required=True)
    parser.add_argument("--dataset-config", action="append", type=Path, default=[])
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    dataset_configs = args.dataset_config or [
        repo_root / "configs" / f"{dataset}.yaml"
        for dataset in ("gender", "age", "rosbank")
    ]
    try:
        result = run_preflight(
            repo_root=repo_root,
            qwen_config=args.qwen_config.resolve(),
            gpt_config=args.gpt_config.resolve(),
            dataset_configs=[path.resolve() for path in dataset_configs],
            environ=os.environ,
            probe=args.probe,
        )
    except (PreflightError, subprocess.CalledProcessError) as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

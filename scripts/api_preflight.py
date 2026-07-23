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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_prompt_pilot import (
    PILOT_SIZE,
    PILOT_SAMPLING_SEED,
    PILOT_VARIANTS,
    stratified_pilot_ids,
)
from scripts.run_model_queue import default_jobs
from src.experiments.artifacts import build_run_manifest, fingerprint
from src.experiments.config_builder import build_runtime_config


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


def validate_prompt_pilot_samples(
    repo_root: Path,
    dataset_config_paths: list[Path],
) -> dict[str, dict[str, Any]]:
    """Materialize all three deterministic validation ID sets in memory only."""
    pandas = importlib.import_module("pandas")
    result: dict[str, dict[str, Any]] = {}
    for config_path in dataset_config_paths:
        config = _load_yaml(config_path)
        dataset = str(config["dataset"]["name"])
        columns = config["dataset"]["columns"]
        validation_path = _resolve_repo_path(
            repo_root, config["dataset"]["splits"]["val"]
        )
        frame = pandas.read_csv(
            validation_path,
            usecols=[
                columns["customer_id"],
                columns["amount"],
                columns["label"],
            ],
        ).rename(
            columns={
                columns["customer_id"]: "customer_id",
                columns["amount"]: "amount",
                columns["label"]: "label",
            }
        )
        identifiers = stratified_pilot_ids(
            frame,
            n_clients=PILOT_SIZE,
            seed=PILOT_SAMPLING_SEED,
        )
        if len(identifiers) != PILOT_SIZE or len(set(identifiers)) != PILOT_SIZE:
            raise PreflightError(
                f"{dataset} pilot selected {len(identifiers)} rows / "
                f"{len(set(identifiers))} unique IDs, expected {PILOT_SIZE}"
            )
        labels = (
            frame[frame["customer_id"].isin(identifiers)]
            .groupby("customer_id")["label"]
            .first()
        )
        if set(labels.unique()) != set(frame["label"].unique()):
            raise PreflightError(
                f"{dataset} pilot does not cover every validation class"
            )
        result[dataset] = {
            "sampling_seed": PILOT_SAMPLING_SEED,
            "selected": len(identifiers),
            "client_ids_sha256": fingerprint(identifiers),
            "variants": list(PILOT_VARIANTS),
        }
    if set(result) != set(EXPECTED_CLIENT_COUNTS):
        raise PreflightError(
            f"Prompt pilot datasets mismatch: {sorted(result)}"
        )
    return result


def planned_runtime_configs(
    *,
    repo_root: Path,
    run_id: str,
    qwen_config: Path,
    gpt_config: Path,
    dataset_configs: list[Path],
) -> list[dict[str, Any]]:
    """Build every possible overnight runtime config without writing it."""
    profiles = {
        str(profile["experiment"]["model_slug"]): profile
        for profile in (_load_yaml(qwen_config), _load_yaml(gpt_config))
    }
    bases = {
        str(base["dataset"]["name"]): base
        for base in (_load_yaml(path) for path in dataset_configs)
    }
    configs: list[dict[str, Any]] = []
    # Every dataset may select any of the three variants.  Validate both final
    # model roots and the Qwen pilot root for every possible selection.
    for dataset in ("gender", "age", "rosbank"):
        pilot_ids_path = (
            Path("logs/runs")
            / run_id
            / "generated"
            / f"{dataset}_pilot_client_ids.json"
        )
        for variant in PILOT_VARIANTS:
            for profile in profiles.values():
                config = build_runtime_config(
                    bases[dataset],
                    profile,
                    run_id=run_id,
                    variant=variant,
                    sampling_seed=PILOT_SAMPLING_SEED,
                    generation_seed=17,
                    claims_seed=17,
                    ml_seed=17,
                    results_root=Path("results/v2"),
                    expected_client_counts=EXPECTED_CLIENT_COUNTS[dataset],
                )
                configs.append(config)
            configs.append(
                build_runtime_config(
                    bases[dataset],
                    profiles["qwen"],
                    run_id=run_id,
                    variant=variant,
                    sampling_seed=PILOT_SAMPLING_SEED,
                    generation_seed=17,
                    claims_seed=17,
                    ml_seed=17,
                    results_root=Path("results/v2/pilot"),
                    client_ids_by_split={"val": str(pilot_ids_path)},
                    expected_client_counts={
                        **EXPECTED_CLIENT_COUNTS[dataset],
                        "val": PILOT_SIZE,
                    },
                )
            )
    return configs


def validate_train_only_and_outputs(
    configs: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, int]:
    checked = 0
    existing = 0
    for config in configs:
        if config.get("pipeline", {}).get("prompt_context_split") != "train":
            raise PreflightError(
                "Runtime config does not enforce a train-only prompt context: "
                f"{config['dataset']['name']}/{config['experiment']['model_slug']}"
            )
        output_root = _resolve_repo_path(repo_root, config["output"]["base_dir"])
        manifest_path = output_root / "manifest.json"
        if output_root.exists():
            existing += 1
            if not manifest_path.is_file():
                raise PreflightError(
                    f"Existing output root has no compatible manifest: {output_root}"
                )
            try:
                current = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise PreflightError(f"Malformed output manifest: {manifest_path}") from exc
            candidate = build_run_manifest(config, repo_root=repo_root)
            if current.get("manifest_sha256") != candidate["manifest_sha256"]:
                raise PreflightError(
                    f"Incompatible existing output root: {output_root}"
                )
        checked += 1
    return {"planned_configs": checked, "existing_compatible_roots": existing}


def validate_queue_matrix(
    *, run_id: str, qwen_config: Path, gpt_config: Path
) -> dict[str, int]:
    profiles = [_load_yaml(qwen_config), _load_yaml(gpt_config)]
    jobs_by_model = {
        profile["experiment"]["model_slug"]: default_jobs(
            profile,
            model_config=qwen_config
            if profile["experiment"]["model_slug"] == "qwen"
            else gpt_config,
            run_id=run_id,
        )
        for profile in profiles
    }
    for model, jobs in jobs_by_model.items():
        full = [job for job in jobs if job.get("stage") == "full"]
        cells = sum(len(job.get("expected_splits", [])) for job in full)
        if len(full) != 3 or cells != 9:
            raise PreflightError(
                f"Queue matrix mismatch for {model}: full_jobs={len(full)}, split_cells={cells}"
            )
        if {job.get("dataset") for job in full} != set(EXPECTED_CLIENT_COUNTS):
            raise PreflightError(f"Queue does not cover all datasets for {model}")
    pilot = [job for job in jobs_by_model["qwen"] if job.get("stage") == "pilot"]
    if (
        len(pilot) != 3
        or {job.get("dataset") for job in pilot} != set(EXPECTED_CLIENT_COUNTS)
        or any(
            job.get("expected_client_counts") != {"val": PILOT_SIZE}
            for job in pilot
        )
    ):
        raise PreflightError(
            "Qwen queue must contain exact 400-client pilots for all datasets"
        )
    return {
        "models": 2,
        "full_jobs": 6,
        "pilot_jobs": 3,
        "split_cells_per_model": 9,
    }


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
    run_id: str = "reviewer-v2",
    expected_counts: dict[str, dict[str, int]] = EXPECTED_CLIENT_COUNTS,
    check_git: bool = True,
    probe: bool = False,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    validate_imports()
    base_url, api_key = _required_environment(environ)
    profiles = validate_model_profiles([qwen_config, gpt_config])
    counts = validate_datasets(repo_root, dataset_configs, expected_counts)
    production_contract = expected_counts == EXPECTED_CLIENT_COUNTS
    if production_contract:
        pilot = validate_prompt_pilot_samples(repo_root, dataset_configs)
        runtime_configs = planned_runtime_configs(
            repo_root=repo_root,
            run_id=run_id,
            qwen_config=qwen_config,
            gpt_config=gpt_config,
            dataset_configs=dataset_configs,
        )
        outputs = validate_train_only_and_outputs(runtime_configs, repo_root=repo_root)
        queue = validate_queue_matrix(
            run_id=run_id,
            qwen_config=qwen_config,
            gpt_config=gpt_config,
        )
    else:
        # Small synthetic fixtures exercise dependency checks without needing a
        # fabricated 400-client historical gender run.
        pilot = {"status": "skipped_nonproduction_fixture"}
        outputs = {"planned_configs": 0, "existing_compatible_roots": 0}
        queue = {"models": 2, "full_jobs": 6, "split_cells_per_model": 9}
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
        "estimated_api_requests": 177_200,
        "prompt_pilots": pilot,
        "runtime_outputs": outputs,
        "queue": queue,
        "probed_models": probed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-id", default="reviewer-v2")
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
            run_id=args.run_id,
            probe=args.probe,
        )
    except (PreflightError, subprocess.CalledProcessError) as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Compose immutable dataset configs with model profiles for reviewer-v2 runs."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

from src.experiments.artifacts import atomic_write_json


EXPECTED_CLIENT_COUNTS: dict[str, dict[str, int]] = {
    "gender": {"train": 6_720, "val": 840, "test": 840},
    "age": {"train": 24_000, "val": 3_000, "test": 3_000},
    "rosbank": {"train": 4_000, "val": 500, "test": 500},
}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_.-")
    if not cleaned:
        raise ValueError("An empty slug is not allowed")
    return cleaned.lower()


def _variant_overlay(variant: str) -> dict[str, Any]:
    variants = {
        "neutral_only": {
            "statistics": {"summary_profile": "legacy_mean_categories"},
            "pipeline": {
                "few_shot_strategy": "random",
                "few_shot_per_class": 2,
                "few_shot_seed": 137,
            },
        },
        "neutral_robust_fewshot": {
            "statistics": {"summary_profile": "robust"},
            "pipeline": {
                "few_shot_strategy": "representative",
                "few_shot_per_class": 2,
                "few_shot_seed": 137,
            },
        },
        "neutral_robust_zero_shot": {
            "statistics": {"summary_profile": "robust"},
            "pipeline": {
                "few_shot_strategy": "representative",
                "few_shot_per_class": 0,
                "few_shot_seed": 137,
            },
        },
        "robust_zero_shot_v2": {
            "statistics": {"summary_profile": "robust"},
            "pipeline": {
                "few_shot_strategy": "representative",
                "few_shot_per_class": 0,
            },
        },
        "legacy_offline": {},
    }
    if variant not in variants:
        raise ValueError(f"Unknown reviewer-v2 variant: {variant}")
    return variants[variant]


def build_runtime_config(
    base_config: dict[str, Any],
    model_profile: dict[str, Any],
    *,
    run_id: str,
    variant: str,
    seed: int | None = None,
    sampling_seed: int | None = None,
    generation_seed: int | None = None,
    claims_seed: int | None = None,
    ml_seed: int | None = None,
    results_root: str | Path = "results/v2",
    client_ids_by_split: dict[str, Any] | None = None,
    expected_client_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return one complete, versioned config consumable by run_pipeline.py."""
    dataset = str(base_config["dataset"]["name"])
    model_slug = slug(model_profile["experiment"]["model_slug"])

    config = deep_merge(base_config, _variant_overlay(variant))
    generation = model_profile.get("generation", {})
    claims_generation = model_profile.get("claims_generation", generation)
    execution = model_profile.get("execution", {})
    profile_experiment = model_profile.get("experiment", {})

    # These seeds control different sources of randomness and must not inherit
    # from one another. In particular, the gender pilot sampling seed (137)
    # must never silently become the LLM decoding seed.
    resolved_sampling_seed = int(
        sampling_seed
        if sampling_seed is not None
        else profile_experiment.get("sampling_seed", 137)
    )
    resolved_generation_seed = int(
        generation_seed
        if generation_seed is not None
        else generation.get(
            "seed",
            profile_experiment.get(
                "generation_seed",
                profile_experiment.get("seed", seed if seed is not None else 17),
            ),
        )
    )
    resolved_claims_seed = int(
        claims_seed
        if claims_seed is not None
        else claims_generation.get(
            "seed",
            profile_experiment.get("claims_seed", resolved_generation_seed),
        )
    )
    resolved_ml_seed = int(
        ml_seed
        if ml_seed is not None
        else profile_experiment.get("ml_seed", 17)
    )
    output_dir = (
        Path(results_root)
        / dataset
        / slug(variant)
        / model_slug
        / f"seed_{resolved_generation_seed}"
    )

    config["experiment"] = {
        **copy.deepcopy(profile_experiment),
        "run_id": run_id,
        "variant": variant,
        # Keep seed as a compatibility alias for consumers that have not yet
        # migrated; it always means the generation seed, never sampling.
        "seed": resolved_generation_seed,
        "sampling_seed": resolved_sampling_seed,
        "generation_seed": resolved_generation_seed,
        "claims_seed": resolved_claims_seed,
        "ml_seed": resolved_ml_seed,
        "seeds": {
            "sampling": resolved_sampling_seed,
            "generation": resolved_generation_seed,
            "claims": resolved_claims_seed,
            "ml": resolved_ml_seed,
        },
        "model_slug": model_slug,
    }
    config["generation"] = copy.deepcopy(generation)
    config["generation"]["seed"] = resolved_generation_seed
    config["claims_generation"] = copy.deepcopy(claims_generation)
    config["claims_generation"]["seed"] = resolved_claims_seed
    config["claims_generation"].setdefault(
        "max_tokens",
        int(config.get("pipeline", {}).get("claims_max_tokens", 2048)),
    )
    config["execution"] = copy.deepcopy(execution)
    config["evaluation"] = deep_merge(
        config.get("evaluation", {}),
        {
            "seeds": [resolved_ml_seed, 101, 947],
            "ml_seed": resolved_ml_seed,
            "bootstrap_seed": resolved_ml_seed,
            "bootstrap_samples": 1000,
        },
    )
    config["clustering"] = deep_merge(
        config.get("clustering", {}),
        model_profile.get("clustering", {}),
    )
    config["feature_selection"] = deep_merge(
        config.get("feature_selection", {}),
        model_profile.get("feature_selection", {}),
    )

    llm = config.setdefault("llm", {})
    llm.update({
        "default_model": generation["model"],
        "temperature": float(generation.get("temperature", llm.get("temperature", 0.8))),
        "top_p": float(generation.get("top_p", llm.get("top_p", 0.9))),
        "seed": resolved_generation_seed,
        "max_concurrent": int(execution.get("initial_concurrency", 64)),
        "batch_size": int(execution.get("initial_concurrency", 64)),
        "initial_concurrency": int(execution.get("initial_concurrency", 64)),
        "fallback_concurrency": int(execution.get("fallback_concurrency", 10)),
        "recovery_clean_batches": int(execution.get("recovery_clean_batches", 10)),
        "cooldown_seconds": float(execution.get("cooldown_seconds", 60)),
        "rate_limit_fallback_concurrent": int(execution.get("fallback_concurrency", 10)),
        "rate_limit_recovery_batches": int(execution.get("recovery_clean_batches", 10)),
        "rate_limit_cooldown_seconds": float(execution.get("cooldown_seconds", 60)),
        "atomic_windows": bool(execution.get("atomic_windows", True)),
        "verify_ssl": bool(execution.get("verify_ssl", True)),
        "events_path": str(output_dir / "events.jsonl"),
    })
    config["pipeline"] = deep_merge(
        config.get("pipeline", {}),
        {
            "claims_model": claims_generation["model"],
            "claims_temperature": float(claims_generation.get("temperature", 0.0)),
            "claims_top_p": float(claims_generation.get("top_p", 1.0)),
            "few_shot_seed": resolved_sampling_seed,
            "sampling_seed": resolved_sampling_seed,
            "generation_seed": resolved_generation_seed,
            "claims_seed": resolved_claims_seed,
            "ml_seed": resolved_ml_seed,
        },
    )
    config["output"]["base_dir"] = str(output_dir)
    if client_ids_by_split:
        config["dataset"]["client_ids_by_split"] = dict(client_ids_by_split)

    split_inputs = {
        str(split): str(path)
        for split, path in config["dataset"].get("splits", {}).items()
    }
    counts = {
        split: int(count)
        for split, count in (
            expected_client_counts
            if expected_client_counts is not None
            else EXPECTED_CLIENT_COUNTS.get(dataset, {})
        ).items()
        if split in split_inputs
    }
    for split, selection in config["dataset"].get("client_ids_by_split", {}).items():
        if split not in split_inputs:
            continue
        if isinstance(selection, (list, tuple, set)):
            counts[split] = len(set(selection))
        elif isinstance(selection, (str, Path)) and Path(selection).is_file():
            selected = json.loads(Path(selection).read_text(encoding="utf-8"))
            if not isinstance(selected, list):
                raise ValueError(f"Expected a JSON list of client IDs in {selection}")
            counts[split] = len(set(selected))

    config["dataset"]["input_paths_by_split"] = split_inputs
    config["dataset"]["expected_client_counts"] = counts
    output_names = {
        "client_stats": "clients_stats",
        "prompts": "prompts",
        "explanations": "explanations",
        "claims": "claims",
    }
    paths_by_split: dict[str, dict[str, str]] = {}
    for split in split_inputs:
        paths_by_split[split] = {}
        for artifact, output_key in output_names.items():
            if output_key not in config["output"]:
                continue
            base = Path(config["output"][output_key])
            suffix = base.suffix or ".jsonl"
            paths_by_split[split][artifact] = str(
                output_dir / f"{base.stem}_{split}{suffix}"
            )
        paths_by_split[split]["direct_metrics"] = str(
            output_dir / f"llm_metrics_{split}.json"
        )
    config["output"]["paths_by_split"] = paths_by_split
    return config


def write_runtime_config(path: str | Path, config: dict[str, Any]) -> Path:
    """Atomically persist YAML and a JSON sidecar useful for inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    atomic_write_json(path.with_suffix(".json"), config)
    return path

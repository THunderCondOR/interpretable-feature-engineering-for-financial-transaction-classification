"""Compose immutable dataset configs with model profiles for reviewer-v2 runs."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from src.experiments.artifacts import atomic_write_json


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
    seed: int,
    results_root: str | Path = "results/v2",
    client_ids_by_split: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return one complete, versioned config consumable by run_pipeline.py."""
    dataset = str(base_config["dataset"]["name"])
    model_slug = slug(model_profile["experiment"]["model_slug"])
    output_dir = Path(results_root) / dataset / slug(variant) / model_slug / f"seed_{int(seed)}"

    config = deep_merge(base_config, _variant_overlay(variant))
    generation = model_profile.get("generation", {})
    claims_generation = model_profile.get("claims_generation", generation)
    execution = model_profile.get("execution", {})

    config["experiment"] = {
        **copy.deepcopy(model_profile.get("experiment", {})),
        "run_id": run_id,
        "variant": variant,
        "seed": int(seed),
        "model_slug": model_slug,
    }
    config["generation"] = copy.deepcopy(generation)
    config["claims_generation"] = copy.deepcopy(claims_generation)
    config["claims_generation"].setdefault(
        "max_tokens",
        int(config.get("pipeline", {}).get("claims_max_tokens", 2048)),
    )
    config["execution"] = copy.deepcopy(execution)
    config["evaluation"] = deep_merge(
        config.get("evaluation", {}),
        {"seeds": [17, 101, 947], "bootstrap_samples": 1000},
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
        "seed": int(generation.get("seed", seed)),
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
            "claims_seed": int(claims_generation.get("seed", seed)),
        },
    )
    config["output"]["base_dir"] = str(output_dir)
    if client_ids_by_split:
        config["dataset"]["client_ids_by_split"] = dict(client_ids_by_split)
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

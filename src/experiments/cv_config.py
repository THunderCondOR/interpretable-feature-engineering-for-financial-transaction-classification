"""Materialize fold-bound pilot and full-run configurations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.experiments.artifacts import file_sha256
from src.experiments.config_builder import build_runtime_config, slug


V5_VARIANTS = (
    "guided_zero_shot_v5",
    "guided_factual_fs1_v5",
    "guided_factual_fs2_v5",
)


def load_fold_manifest(
    prepared_root: Path,
    dataset: str,
    protocol: str,
    fold: int,
) -> tuple[Path, dict[str, Any]]:
    path = (
        prepared_root / dataset / protocol / f"fold_{int(fold)}"
        / "fold_manifest.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("dataset") != dataset
        or payload.get("protocol") != protocol
        or int(payload.get("fold", -1)) != int(fold)
    ):
        raise ValueError(f"Incompatible fold manifest: {path}")
    return path, payload


def _id_path(manifest_path: Path, role: str) -> str:
    path = manifest_path.parent / f"{role}_ids.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def build_cv_runtime_config(
    base_config: dict[str, Any],
    model_profile: dict[str, Any],
    *,
    run_id: str,
    variant: str,
    fold_manifest_path: Path,
    fold_manifest: dict[str, Any],
    benchmark_manifest_path: Path,
    results_root: Path,
    mode: str,
) -> dict[str, Any]:
    """Build a config whose train-only boundary is explicit and hash-bound."""
    if mode not in {"pilot", "full"}:
        raise ValueError(f"Unknown CV config mode: {mode}")
    dataset = str(fold_manifest["dataset"])
    protocol = str(fold_manifest["protocol"])
    fold = int(fold_manifest["fold"])
    benchmark = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    events_path = str(Path(benchmark["events"]))
    if mode == "pilot":
        roles = {"train": "inner_train", "val": "inner_validation"}
    else:
        roles = {"train": "outer_train", "test": "outer_test"}
    configured_base = copy.deepcopy(base_config)
    configured_base["dataset"]["splits"] = {
        split: events_path for split in roles
    }
    filters = {
        split: _id_path(fold_manifest_path, role)
        for split, role in roles.items()
    }
    counts = {
        split: int(fold_manifest["counts"][role])
        for split, role in roles.items()
    }
    config = build_runtime_config(
        configured_base,
        model_profile,
        run_id=run_id,
        variant=variant,
        results_root=results_root,
        client_ids_by_split=filters,
        expected_client_counts=counts,
    )
    model_slug = slug(config["experiment"]["model_slug"])
    output_root = (
        results_root / dataset / protocol / f"fold_{fold}"
        / ("pilot" if mode == "pilot" else variant)
        / model_slug / f"seed_{config['experiment']['generation_seed']}"
    )
    if mode == "pilot":
        output_root = output_root / variant
    config["output"]["base_dir"] = str(output_root)
    for split, paths in config["output"]["paths_by_split"].items():
        for artifact, old_path in list(paths.items()):
            old = Path(old_path)
            paths[artifact] = str(output_root / old.name)
    config["pipeline"].update({
        "prompt_context_split": "train",
        "prompt_context_population": "full_train_split",
        "prompt_context_apply_client_filter": True,
        "cv_mode": mode,
    })
    config["dataset"]["expected_client_counts"] = counts
    config["cv"] = {
        "schema_version": 1,
        "dataset": dataset,
        "protocol": protocol,
        "fold": fold,
        "mode": mode,
        "split_roles": roles,
        "fold_manifest": str(fold_manifest_path),
        "fold_manifest_sha256": file_sha256(fold_manifest_path),
        "fold_signature": fold_manifest["fold_signature"],
        "benchmark_manifest": str(benchmark_manifest_path),
        "benchmark_manifest_sha256": file_sha256(benchmark_manifest_path),
        "outer_test_labels_forbidden_during_fit": True,
    }
    config["experiment"].update({
        "fold": fold,
        "protocol": protocol,
        "fold_signature": fold_manifest["fold_signature"],
    })
    return config

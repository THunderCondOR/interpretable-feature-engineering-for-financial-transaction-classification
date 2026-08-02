"""Manifest-bound runtime configuration for isolated benchmarks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.experiments.artifacts import file_sha256
from src.experiments.config_builder import build_runtime_config, slug


VARIANTS = (
    "guided_zero_shot_v5",
    "guided_factual_fs1_v5",
    "guided_factual_fs2_v5",
)


def load_benchmark_manifest(path: Path, expected_dataset: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != expected_dataset or not payload.get("manifest_signature"):
        raise ValueError(f"Incompatible isolated benchmark manifest: {path}")
    return payload


def build_isolated_runtime_config(
    base_config: dict[str, Any],
    model_profile: dict[str, Any],
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    run_id: str,
    variant: str,
    results_root: Path,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"pilot", "full"}:
        raise ValueError(f"Unknown isolated runtime mode: {mode}")
    dataset = str(manifest["dataset"])
    roles = (
        {"train": "train", "val": "pilot_val"}
        if mode == "pilot"
        else {"train": "train", "val": "val", "test": "test"}
    )
    configured = copy.deepcopy(base_config)
    configured["dataset"]["splits"] = {
        split: str(manifest["events"]) for split in roles
    }
    filters = {split: str(manifest["roles"][role]) for split, role in roles.items()}
    counts = {split: int(manifest["counts"][role]) for split, role in roles.items()}
    config = build_runtime_config(
        configured,
        model_profile,
        run_id=run_id,
        variant=variant,
        sampling_seed=137,
        generation_seed=17,
        claims_seed=17,
        ml_seed=17,
        results_root=results_root,
        client_ids_by_split=filters,
        expected_client_counts=counts,
    )
    model_slug = slug(config["experiment"]["model_slug"])
    output_root = (
        results_root / "runs" / slug(run_id) / dataset / str(manifest["protocol"])
        / mode / variant / model_slug / "seed_17"
    )
    config["output"]["base_dir"] = str(output_root)
    for paths in config["output"]["paths_by_split"].values():
        for artifact, old_path in list(paths.items()):
            paths[artifact] = str(output_root / Path(old_path).name)
    config["pipeline"].update({
        "prompt_context_split": "train",
        "prompt_context_population": "full_train_split",
        "prompt_context_apply_client_filter": True,
        "isolated_benchmark_mode": mode,
    })
    config["dataset"]["expected_client_counts"] = counts
    config["isolated_benchmark"] = {
        "schema_version": 1,
        "dataset": dataset,
        "protocol": manifest["protocol"],
        "mode": mode,
        "split_roles": roles,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_signature": manifest["manifest_signature"],
        "val_test_labels_forbidden_during_feature_fit": True,
        # Git HEAD alone is insufficient when a reviewed but not-yet-committed
        # launch is required. Pin the exact implementation bytes consumed by
        # the paid generation so provenance remains honest either way.
        "implementation_files_sha256": {
            str(path): file_sha256(path)
            for path in (
                Path("run_pipeline.py"),
                Path("src/data/profiles.py"),
                Path("src/pipeline/prompt_builder.py"),
                Path("src/pipeline/explanation_gen.py"),
                Path("src/pipeline/claims_extractor.py"),
                Path("src/utils/async_api.py"),
                Path("src/benchmarks/runtime.py"),
                manifest_path,
            )
        },
    }
    config["experiment"].update({
        "protocol": manifest["protocol"],
        "benchmark_manifest_signature": manifest["manifest_signature"],
    })
    return config

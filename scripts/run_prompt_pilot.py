#!/usr/bin/env python3
"""Run controlled zero-shot/FS1/FS2 prompt pilots for any supported dataset.

The command is read-only unless ``--execute`` is supplied.  API stages require
both ``--execute-api`` and ``--until-complete``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import add_features, load_dataset
from src.data.client_sampling import stratified_client_ids
from src.evaluation.prompt_pilot import (
    AGE_LABEL_SEMANTICS,
    PILOT_VARIANTS,
    ZERO_SHOT,
    age_interpretation_diagnostics,
    paired_balanced_accuracy_delta,
    rationale_diagnostics,
    select_age_label_semantics,
    select_prompt_variant,
)
from src.experiments.artifacts import (
    atomic_write_json,
    file_sha256,
    fingerprint,
    git_revision,
)
from src.experiments.config_builder import (
    EXPECTED_CLIENT_COUNTS,
    build_runtime_config,
    load_yaml,
    slug,
    write_runtime_config,
)


PILOT_SAMPLING_SEED = 137
PILOT_SIZE = 400


def stratified_pilot_ids(
    transactions: pd.DataFrame,
    *,
    n_clients: int = PILOT_SIZE,
    seed: int = PILOT_SAMPLING_SEED,
) -> list[int]:
    """Sample validation clients by label, activity, and absolute volume."""
    return stratified_client_ids(
        transactions,
        n_clients=n_clients,
        seed=seed,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def _exact_rows(path: Path, expected_ids: list[int]) -> list[dict[str, Any]]:
    expected = set(expected_ids)
    rows = _read_jsonl(path)
    observed = [int(row.get("customer_id", -1)) for row in rows]
    duplicates = [
        customer_id
        for customer_id, count in Counter(observed).items()
        if count > 1
    ]
    if duplicates or set(observed) != expected:
        raise ValueError(
            f"Pilot coverage mismatch in {path}: duplicates={len(duplicates)}, "
            f"missing={len(expected - set(observed))}, "
            f"unexpected={len(set(observed) - expected)}"
        )
    return sorted(rows, key=lambda row: int(row["customer_id"]))


def _config_artifact(config: dict[str, Any], stem: str) -> Path:
    return Path(config["output"]["base_dir"]) / f"{stem}_val.jsonl"


def validate_pilot_outputs(
    config: dict[str, Any],
    expected_ids: list[int],
) -> dict[str, Any]:
    explanations = _exact_rows(
        _config_artifact(config, "explanations"), expected_ids
    )
    prompts = _exact_rows(_config_artifact(config, "prompts"), expected_ids)
    invalid = [
        int(row["customer_id"])
        for row in explanations
        if row.get("error")
        or row.get("predicted") is None
        or not str(row.get("explanation", "")).strip()
    ]
    if invalid:
        raise RuntimeError(f"Pilot has {len(invalid)} incomplete explanations")
    output = Path(config["output"]["base_dir"])
    metrics_path = output / "llm_metrics_val.json"
    telemetry_path = output / "prompt_length_telemetry_val.json"
    manifest_path = output / "manifest.json"
    for path in (metrics_path, telemetry_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        int(metrics.get("n_rows", -1)) != len(expected_ids)
        or int(metrics.get("n_scored", -1)) != len(expected_ids)
        or float(metrics.get("coverage", 0.0)) != 1.0
    ):
        raise RuntimeError(f"Incomplete pilot metrics: {metrics_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != config["experiment"]["run_id"]:
        raise RuntimeError(f"Pilot manifest run mismatch: {manifest_path}")
    return {
        "metrics": metrics,
        "explanations": explanations,
        "prompts": prompts,
        "prompt_length_telemetry": json.loads(
            telemetry_path.read_text(encoding="utf-8")
        ),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }


def materialize(
    *,
    dataset: str,
    run_id: str,
    model_config: Path,
    gpt_model_config: Path,
    sample_size: int,
    sampling_seed: int,
    generated_dir: Path,
    results_root: Path,
) -> dict[str, Any]:
    base_path = Path("configs") / f"{dataset}.yaml"
    base = load_yaml(base_path)
    profile = load_yaml(model_config)
    validation = add_features(load_dataset(base, "val"))
    pilot_ids = stratified_pilot_ids(
        validation,
        n_clients=sample_size,
        seed=sampling_seed,
    )
    if len(pilot_ids) != sample_size:
        raise ValueError(
            f"{dataset} pilot requires {sample_size} validation clients; "
            f"dataset supplied {len(pilot_ids)}"
        )
    generated_dir.mkdir(parents=True, exist_ok=True)
    ids_path = generated_dir / f"{dataset}_pilot_client_ids.json"
    atomic_write_json(ids_path, pilot_ids)
    configs: dict[str, str] = {}
    hashes: dict[str, str] = {}
    semantics_values = AGE_LABEL_SEMANTICS if dataset == "age" else ("standard",)
    cells: dict[str, dict[str, str]] = {}
    for label_semantics in semantics_values:
        for variant in PILOT_VARIANTS:
            cell = (
                variant
                if label_semantics == "standard"
                else f"{label_semantics}::{variant}"
            )
            config = build_runtime_config(
                base,
                profile,
                run_id=run_id,
                variant=variant,
                label_semantics=label_semantics,
                sampling_seed=sampling_seed,
                results_root=results_root / "pilot",
                client_ids_by_split={"val": str(ids_path)},
                expected_client_counts={
                    **EXPECTED_CLIENT_COUNTS[dataset],
                    "val": sample_size,
                },
            )
            path = (
                generated_dir
                / f"{dataset}_{label_semantics}_{variant}_qwen.yaml"
            )
            write_runtime_config(path, config)
            configs[cell] = str(path)
            hashes[cell] = file_sha256(path)
            cells[cell] = {
                "prompt_format": variant,
                "label_semantics": label_semantics,
            }
    payload = {
        "run_id": run_id,
        "dataset": dataset,
        "sample_size": sample_size,
        "sampling_seed": sampling_seed,
        "pilot_ids": str(ids_path),
        "pilot_ids_sha256": file_sha256(ids_path),
        "variants": list(PILOT_VARIANTS),
        "label_semantics": list(semantics_values),
        "pilot_cells": cells,
        "pilot_configs": configs,
        "pilot_config_sha256": hashes,
        "model_config": str(model_config),
        "model_config_sha256": file_sha256(model_config),
        "gpt_model_config": str(gpt_model_config),
        "gpt_model_config_sha256": file_sha256(gpt_model_config),
        "base_config": str(base_path),
        "base_config_sha256": file_sha256(base_path),
    }
    payload["materialization_sha256"] = fingerprint(payload)
    path = generated_dir / f"{dataset}_pilot.materialized.json"
    atomic_write_json(path, payload)
    return payload


def _pipeline_command(config_path: str) -> list[str]:
    return [
        sys.executable,
        "run_pipeline.py",
        "--config",
        config_path,
        "--steps",
        "stats,prompts,cot,llm_eval",
        "--splits",
        "val",
        "--execute",
        "--until-complete",
    ]


def run_pilot(materialized: dict[str, Any]) -> Path:
    ids = json.loads(
        Path(materialized["pilot_ids"]).read_text(encoding="utf-8")
    )
    manifests: dict[str, str] = {}
    for cell, config_path in materialized["pilot_configs"].items():
        if file_sha256(config_path) != materialized["pilot_config_sha256"][cell]:
            raise RuntimeError(f"Pilot config changed: {config_path}")
        subprocess.run(
            _pipeline_command(config_path),
            cwd=REPO_ROOT,
            check=True,
        )
        evidence = validate_pilot_outputs(load_yaml(config_path), ids)
        manifests[cell] = str(evidence["manifest_sha256"])
    marker = {
        "run_id": materialized["run_id"],
        "dataset": materialized["dataset"],
        "model_slug": "qwen",
        "variant": "controlled_prompt_pilot_v4",
        "variants": list(PILOT_VARIANTS),
        "label_semantics": materialized["label_semantics"],
        "pilot_cells": materialized["pilot_cells"],
        "splits": ["val"],
        "expected_counts": {"val": materialized["sample_size"]},
        "pilot_ids_sha256": materialized["pilot_ids_sha256"],
        "pilot_config_sha256": materialized["pilot_config_sha256"],
        "manifest_sha256": manifests,
        "status": "completed",
    }
    path = (
        Path("logs/runs")
        / materialized["run_id"]
        / "completion"
        / f"qwen_{materialized['dataset']}_pilot.json"
    )
    atomic_write_json(path, marker)
    return path


def select_variant(
    materialized: dict[str, Any],
    *,
    generated_dir: Path,
    results_root: Path,
) -> dict[str, Any]:
    marker_path = (
        Path("logs/runs")
        / materialized["run_id"]
        / "completion"
        / f"qwen_{materialized['dataset']}_pilot.json"
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("run_id") != materialized["run_id"]
        or marker.get("pilot_ids_sha256") != materialized["pilot_ids_sha256"]
        or marker.get("status") != "completed"
    ):
        raise RuntimeError(f"Stale or incompatible pilot completion: {marker_path}")
    ids = json.loads(
        Path(materialized["pilot_ids"]).read_text(encoding="utf-8")
    )
    evidence: dict[str, dict[str, Any]] = {}
    for cell, config_path in materialized["pilot_configs"].items():
        if file_sha256(config_path) != materialized["pilot_config_sha256"][cell]:
            raise RuntimeError(f"Pilot config changed: {config_path}")
        evidence[cell] = validate_pilot_outputs(load_yaml(config_path), ids)
    metrics = {cell: item["metrics"] for cell, item in evidence.items()}
    diagnostics = {
        cell: rationale_diagnostics(
            item["explanations"], item["prompts"]
        )
        for cell, item in evidence.items()
    }
    if materialized["dataset"] == "age":
        format_decisions: dict[str, dict[str, Any]] = {}
        format_deltas: dict[str, dict[str, dict[str, Any]]] = {}
        for semantics in AGE_LABEL_SEMANTICS:
            zero_cell = f"{semantics}::{ZERO_SHOT}"
            semantics_metrics = {
                variant: metrics[f"{semantics}::{variant}"]
                for variant in PILOT_VARIANTS
            }
            semantics_deltas = {
                variant: paired_balanced_accuracy_delta(
                    evidence[zero_cell]["explanations"],
                    evidence[f"{semantics}::{variant}"]["explanations"],
                    seed=materialized["sampling_seed"],
                )
                for variant in PILOT_VARIANTS
                if variant != ZERO_SHOT
            }
            format_deltas[semantics] = semantics_deltas
            format_decisions[semantics] = select_prompt_variant(
                semantics_metrics, semantics_deltas
            )
        opaque_variant = format_decisions["age_opaque"]["selected_variant"]
        ordered_variant = format_decisions["age_ordered"]["selected_variant"]
        opaque_cell = f"age_opaque::{opaque_variant}"
        ordered_cell = f"age_ordered::{ordered_variant}"
        semantics_delta = paired_balanced_accuracy_delta(
            evidence[opaque_cell]["explanations"],
            evidence[ordered_cell]["explanations"],
            seed=materialized["sampling_seed"],
        )
        semantics_decision = select_age_label_semantics(
            opaque_variant=opaque_cell,
            ordered_variant=ordered_cell,
            metrics=metrics,
            ordered_vs_opaque=semantics_delta,
        )
        selected_cell = semantics_decision["selected_variant"]
        selected_semantics, selected = selected_cell.split("::", 1)
        decision = {
            "selected_variant": selected,
            "selected_label_semantics": selected_semantics,
            "format_decisions": format_decisions,
            "format_paired_balanced_accuracy_deltas": format_deltas,
            "label_semantics_decision": semantics_decision,
            "ordered_vs_opaque_paired_delta": semantics_delta,
        }
        age_diagnostics = {
            cell: age_interpretation_diagnostics(item["explanations"])
            for cell, item in evidence.items()
        }
    else:
        deltas = {
            variant: paired_balanced_accuracy_delta(
                evidence[ZERO_SHOT]["explanations"],
                evidence[variant]["explanations"],
                seed=materialized["sampling_seed"],
            )
            for variant in PILOT_VARIANTS
            if variant != ZERO_SHOT
        }
        decision = select_prompt_variant(metrics, deltas)
        selected = decision["selected_variant"]
        selected_semantics = "standard"
        selected_cell = selected
        decision["selected_label_semantics"] = selected_semantics
        decision["format_paired_balanced_accuracy_deltas"] = deltas
        age_diagnostics = {}
    base = load_yaml(materialized["base_config"])
    selected_configs: dict[str, dict[str, str]] = {}
    for model_path in (
        Path(materialized["model_config"]),
        Path(materialized["gpt_model_config"]),
    ):
        profile = load_yaml(model_path)
        model_slug = slug(profile["experiment"]["model_slug"])
        config = build_runtime_config(
            base,
            profile,
            run_id=materialized["run_id"],
            variant=selected,
            label_semantics=selected_semantics,
            sampling_seed=materialized["sampling_seed"],
            results_root=results_root,
            expected_client_counts=EXPECTED_CLIENT_COUNTS[
                materialized["dataset"]
            ],
        )
        path = (
            generated_dir
            / f"{materialized['dataset']}_selected_{model_slug}.yaml"
        )
        write_runtime_config(path, config)
        selected_configs[model_slug] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "source_model_config": str(model_path),
            "source_model_config_sha256": file_sha256(model_path),
        }
    selection = {
        "status": "completed",
        "run_id": materialized["run_id"],
        "dataset": materialized["dataset"],
        **decision,
        "selection_split": "val",
        "pilot_ids": materialized["pilot_ids"],
        "pilot_ids_sha256": materialized["pilot_ids_sha256"],
        "metrics": metrics,
        "diagnostics": diagnostics,
        "age_interpretation_diagnostics": age_diagnostics,
        "prompt_length_telemetry": {
            variant: item["prompt_length_telemetry"]
            for variant, item in evidence.items()
        },
        "selected_configs": selected_configs,
        "git_revision": git_revision(REPO_ROOT),
    }
    selection["selection_sha256"] = fingerprint(selection)
    atomic_write_json(
        generated_dir / f"{materialized['dataset']}_selection.json",
        selection,
    )
    return selection


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    semantics_count = 2 if args.dataset == "age" else 1
    return {
        "mode": "execute" if args.execute else "dry-run",
        "run_id": args.run_id,
        "dataset": args.dataset,
        "stage": args.stage,
        "sample_size": args.sample_size,
        "sampling_seed": args.sampling_seed,
        "variants": list(args.variants),
        "label_semantics": (
            list(AGE_LABEL_SEMANTICS) if args.dataset == "age" else ["standard"]
        ),
        "selection_rule": (
            "FS must improve balanced accuracy by strictly more than 0.02 "
            "and have paired bootstrap CI lower bound > 0; FS1 wins ties <= 0.005"
        ),
        "api_requests": args.sample_size * len(args.variants) * semantics_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=EXPECTED_CLIENT_COUNTS)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument(
        "--gpt-model-config",
        type=Path,
        default=Path("configs/v2/gpt_oss.yaml"),
    )
    parser.add_argument("--run-id", default="reviewer-v4-english-20260723")
    parser.add_argument("--sample-size", type=int, default=PILOT_SIZE)
    parser.add_argument("--sampling-seed", type=int, default=PILOT_SAMPLING_SEED)
    parser.add_argument(
        "--variants",
        default=",".join(PILOT_VARIANTS),
        type=lambda value: tuple(part.strip() for part in value.split(",") if part.strip()),
    )
    parser.add_argument(
        "--stage",
        choices=("plan", "materialize", "pilot", "select", "all"),
        default="all",
    )
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/v2"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    if tuple(args.variants) != PILOT_VARIANTS:
        raise ValueError(
            "Controlled v4 pilot requires exactly: " + ",".join(PILOT_VARIANTS)
        )
    if args.sample_size < 1:
        raise ValueError("--sample-size must be positive")
    print(json.dumps(build_plan(args), ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if args.stage in {"pilot", "all"} and (
        not args.execute_api or not args.until_complete
    ):
        raise ValueError(
            "Pilot API execution requires --execute-api and --until-complete"
        )
    generated_dir = args.generated_dir or (
        Path("logs/runs") / args.run_id / "generated"
    )
    materialized = materialize(
        dataset=args.dataset,
        run_id=args.run_id,
        model_config=args.model_config,
        gpt_model_config=args.gpt_model_config,
        sample_size=args.sample_size,
        sampling_seed=args.sampling_seed,
        generated_dir=generated_dir,
        results_root=args.results_root,
    )
    if args.stage in {"plan", "materialize"}:
        return
    if args.stage in {"pilot", "all"}:
        marker = run_pilot(materialized)
        print(f"Pilot completion -> {marker}")
    if args.stage in {"select", "all"}:
        selection = select_variant(
            materialized,
            generated_dir=generated_dir,
            results_root=args.results_root,
        )
        print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

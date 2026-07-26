#!/usr/bin/env python3
"""Select a prompt independently inside every outer benchmark fold."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.benchmark_registry import BENCHMARKS
from src.data.entity_ids import canonical_entity_id, entity_sort_key
from src.evaluation.prompt_pilot import (
    paired_primary_metric_delta,
    rationale_diagnostics,
    select_prompt_variant_by_metric,
)
from src.experiments.artifacts import (
    atomic_write_json,
    file_sha256,
    fingerprint,
)
from src.experiments.config_builder import load_yaml, write_runtime_config
from src.experiments.cv_config import (
    V5_VARIANTS,
    build_cv_runtime_config,
    load_fold_manifest,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL {path}:{line_number}") from exc
    return rows


def exact_success_rows(
    path: Path,
    expected_ids: set[int | str],
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    observed: dict[int | str, dict[str, Any]] = {}
    for row in rows:
        entity_id = canonical_entity_id(row["customer_id"])
        if entity_id in observed:
            raise ValueError(f"Duplicate ID {entity_id!r} in {path}")
        if (
            row.get("error")
            or row.get("predicted") is None
            or not str(row.get("explanation", "")).strip()
        ):
            raise ValueError(f"Incomplete pilot record {entity_id!r} in {path}")
        observed[entity_id] = row
    if set(observed) != expected_ids:
        raise ValueError(
            f"Pilot coverage mismatch in {path}: "
            f"missing={len(expected_ids - set(observed))}, "
            f"unexpected={len(set(observed) - expected_ids)}"
        )
    return [
        observed[value] for value in sorted(expected_ids, key=entity_sort_key)
    ]


def _pipeline_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "run_pipeline.py",
        "--config",
        str(config_path),
        "--steps",
        "stats,prompts,cot,llm_eval",
        "--splits",
        "val",
        "--execute",
        "--until-complete",
    ]


def materialize_fold(
    *,
    dataset: str,
    fold: int,
    run_id: str,
    prepared_root: Path,
    results_root: Path,
    generated_root: Path,
    qwen_profile: dict[str, Any],
    gpt_profile: dict[str, Any],
) -> dict[str, Any]:
    spec = BENCHMARKS[dataset]
    fold_path, fold_manifest = load_fold_manifest(
        prepared_root, dataset, spec.protocol, fold
    )
    benchmark_path = (
        prepared_root / dataset / spec.protocol / "benchmark_manifest.json"
    )
    base_path = Path("configs/v5") / f"{dataset}.yaml"
    base = load_yaml(base_path)
    cell_root = generated_root / dataset / f"fold_{fold}"
    cell_root.mkdir(parents=True, exist_ok=True)
    pilot_configs = {}
    for variant in V5_VARIANTS:
        config = build_cv_runtime_config(
            base,
            qwen_profile,
            run_id=run_id,
            variant=variant,
            fold_manifest_path=fold_path,
            fold_manifest=fold_manifest,
            benchmark_manifest_path=benchmark_path,
            results_root=results_root,
            mode="pilot",
        )
        path = cell_root / f"pilot_{variant}.yaml"
        write_runtime_config(path, config)
        pilot_configs[variant] = str(path)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": dataset,
        "protocol": spec.protocol,
        "fold": fold,
        "primary_metric": spec.primary_metric,
        "pilot_size": int(fold_manifest["counts"]["inner_validation"]),
        "pilot_ids_sha256": fold_manifest["id_hashes"]["inner_validation"],
        "fold_signature": fold_manifest["fold_signature"],
        "fold_manifest": str(fold_path),
        "benchmark_manifest": str(benchmark_path),
        "base_config": str(base_path),
        "pilot_configs": pilot_configs,
        "pilot_config_sha256": {
            variant: file_sha256(path)
            for variant, path in pilot_configs.items()
        },
        "qwen_profile": qwen_profile,
        "gpt_profile": gpt_profile,
    }
    payload["materialization_signature"] = fingerprint(payload)
    atomic_write_json(cell_root / "pilot_materialization.json", payload)
    return payload


def run_fold_pilot(materialized: dict[str, Any]) -> None:
    for variant in V5_VARIANTS:
        path = Path(materialized["pilot_configs"][variant])
        if file_sha256(path) != materialized["pilot_config_sha256"][variant]:
            raise RuntimeError(f"Pilot config changed after materialization: {path}")
        subprocess.run(_pipeline_command(path), cwd=REPO_ROOT, check=True)


def select_fold(
    materialized: dict[str, Any],
    *,
    results_root: Path,
    generated_root: Path,
) -> dict[str, Any]:
    fold_manifest = json.loads(
        Path(materialized["fold_manifest"]).read_text(encoding="utf-8")
    )
    expected_ids = {
        canonical_entity_id(value)
        for value in fold_manifest["ids"]["inner_validation"]
    }
    metrics: dict[str, dict[str, Any]] = {}
    explanations: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for variant, config_path in materialized["pilot_configs"].items():
        config = load_yaml(config_path)
        root = Path(config["output"]["base_dir"])
        metrics[variant] = json.loads(
            (root / "llm_metrics_val.json").read_text(encoding="utf-8")
        )
        explanations[variant] = exact_success_rows(
            root / "explanations_val.jsonl", expected_ids
        )
        prompts = read_jsonl(root / "prompts_val.jsonl")
        diagnostics[variant] = rationale_diagnostics(
            explanations[variant], prompts
        )
    zero = "guided_zero_shot_v5"
    paired = {
        variant: paired_primary_metric_delta(
            explanations[zero],
            explanations[variant],
            metric=materialized["primary_metric"],
            seed=137 + int(materialized["fold"]),
        )
        for variant in V5_VARIANTS
        if variant != zero
    }
    decision = select_prompt_variant_by_metric(
        metrics,
        paired,
        metric=materialized["primary_metric"],
    )
    selected = decision["selected_variant"]
    base = load_yaml(materialized["base_config"])
    fold_path = Path(materialized["fold_manifest"])
    benchmark_path = Path(materialized["benchmark_manifest"])
    cell_root = generated_root / materialized["dataset"] / (
        f"fold_{materialized['fold']}"
    )
    configs = {}
    for name, profile in (
        ("qwen", materialized["qwen_profile"]),
        ("gpt_oss", materialized["gpt_profile"]),
    ):
        config = build_cv_runtime_config(
            base,
            profile,
            run_id=materialized["run_id"],
            variant=selected,
            fold_manifest_path=fold_path,
            fold_manifest=fold_manifest,
            benchmark_manifest_path=benchmark_path,
            results_root=results_root,
            mode="full",
        )
        path = cell_root / f"selected_{name}.yaml"
        write_runtime_config(path, config)
        configs[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "output_root": config["output"]["base_dir"],
        }
    selection = {
        "schema_version": 1,
        "status": "completed",
        "run_id": materialized["run_id"],
        "dataset": materialized["dataset"],
        "protocol": materialized["protocol"],
        "fold": materialized["fold"],
        "fold_signature": materialized["fold_signature"],
        "pilot_ids_sha256": materialized["pilot_ids_sha256"],
        "primary_metric": materialized["primary_metric"],
        "metrics": metrics,
        "paired_deltas": paired,
        "diagnostics": diagnostics,
        **decision,
        "selected_configs": configs,
        "full_expected_client_counts": {
            "train": int(fold_manifest["counts"]["outer_train"]),
            "test": int(fold_manifest["counts"]["outer_test"]),
        },
    }
    selection["selection_signature"] = fingerprint(selection)
    selection["selection_sha256"] = fingerprint(selection)
    atomic_write_json(cell_root / "prompt_selection.json", selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--run-id", default="reviewer-v5-benchmarks")
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("data/benchmarks_v5")
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/v5"))
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=Path("logs/runs/reviewer-v5-benchmarks/generated"),
    )
    parser.add_argument(
        "--qwen-config", type=Path, default=Path("configs/v2/qwen.yaml")
    )
    parser.add_argument(
        "--gpt-config", type=Path, default=Path("configs/v2/gpt_oss.yaml")
    )
    parser.add_argument(
        "--stage",
        choices=("all", "materialize", "pilot", "select"),
        default="all",
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    generated_root = (
        args.generated_root
        if args.generated_root != Path("logs/runs/reviewer-v5-benchmarks/generated")
        else Path("logs/runs") / args.run_id / "generated"
    )
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "dataset": args.dataset,
        "protocol": BENCHMARKS[args.dataset].protocol,
        "folds": folds,
        "variants": list(V5_VARIANTS),
        "primary_metric": BENCHMARKS[args.dataset].primary_metric,
        "pilot_clients_per_fold": (
            400 if args.dataset == "datafusion_education" else 100
        ),
        "stage": args.stage,
        "api_required": args.stage in {"all", "pilot"},
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if args.stage in {"all", "pilot"} and not (
        args.execute_api and args.until_complete
    ):
        raise ValueError("Pilot API stages require --execute-api --until-complete")
    qwen, gpt = load_yaml(args.qwen_config), load_yaml(args.gpt_config)
    for fold in folds:
        materialized_path = (
            generated_root / args.dataset / f"fold_{fold}"
            / "pilot_materialization.json"
        )
        if args.stage in {"all", "materialize"}:
            materialized = materialize_fold(
                dataset=args.dataset,
                fold=fold,
                run_id=args.run_id,
                prepared_root=args.prepared_root,
                results_root=args.results_root,
                generated_root=generated_root,
                qwen_profile=qwen,
                gpt_profile=gpt,
            )
        else:
            materialized = json.loads(
                materialized_path.read_text(encoding="utf-8")
            )
        if args.stage in {"all", "pilot"}:
            run_fold_pilot(materialized)
        if args.stage in {"all", "select"}:
            select_fold(
                materialized,
                results_root=args.results_root,
                generated_root=generated_root,
            )


if __name__ == "__main__":
    main()

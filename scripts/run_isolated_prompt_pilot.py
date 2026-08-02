#!/usr/bin/env python3
"""Materialize, audit, run, and select prompts for one isolated benchmark."""

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

from scripts.audit_isolated_prompts import audit
from src.benchmarks.runtime import VARIANTS, build_isolated_runtime_config, load_benchmark_manifest
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
    prompt_signature,
)
from src.experiments.config_builder import load_yaml, slug, write_runtime_config
from src.pipeline.llm_eval import evaluate_llm_predictions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL {path}:{line_number}") from exc
    return rows


def exact_success_rows(path: Path, expected_ids: set[int | str]) -> list[dict[str, Any]]:
    observed = {}
    for row in read_jsonl(path):
        entity_id = canonical_entity_id(row["customer_id"])
        if entity_id in observed:
            raise ValueError(f"Duplicate pilot ID {entity_id!r} in {path}")
        if row.get("error") or row.get("predicted") is None or not str(row.get("explanation", "")).strip():
            raise ValueError(f"Incomplete pilot row {entity_id!r} in {path}")
        observed[entity_id] = row
    if set(observed) != expected_ids:
        raise ValueError(
            f"Pilot coverage mismatch in {path}: missing={len(expected_ids-set(observed))}, "
            f"unexpected={len(set(observed)-expected_ids)}"
        )
    return [observed[value] for value in sorted(expected_ids, key=entity_sort_key)]


def generated_cell(generated_root: Path, run_id: str, dataset: str) -> Path:
    return generated_root / slug(run_id) / dataset


def reuse_compatible_pilot_generation(
    config: dict[str, Any], expected_ids: set[int | str]
) -> bool:
    """Reuse generation when only downstream evaluator code changed.

    This deliberately does not trust the run-level manifest.  It recomputes
    every prompt-generation signature from the current config and requires
    exact pilot-ID coverage and prompt provenance before metrics are rebuilt.
    """
    root = Path(config["output"]["base_dir"])
    prompts_path = root / "prompts_val.jsonl"
    explanations_path = root / "explanations_val.jsonl"
    if not prompts_path.is_file() or not explanations_path.is_file():
        return False
    prompts = {
        canonical_entity_id(row["customer_id"]): row
        for row in read_jsonl(prompts_path)
    }
    if set(prompts) != expected_ids or len(prompts) != len(expected_ids):
        return False
    try:
        explanations = exact_success_rows(explanations_path, expected_ids)
    except (KeyError, TypeError, ValueError):
        return False

    llm = dict(config["llm"])
    llm.update(config.get("execution", {}))
    llm.update(config.get("generation", {}))
    model = str(llm.get("model", llm["default_model"]))
    decoding = {
        "temperature": llm.get("temperature", 1.0),
        "top_p": llm.get("top_p", 0.9),
        "max_tokens": llm.get("max_tokens", 2048),
        "seed": llm.get("seed"),
        "extra_body": llm.get("extra_body"),
    }
    for row in explanations:
        entity_id = canonical_entity_id(row["customer_id"])
        prompt = prompts[entity_id]
        expected_signature = prompt_signature(
            system_prompt=prompt["system_prompt"],
            user_prompt=prompt["user_prompt"],
            model=model,
            decoding=decoding,
            sample_id=int(row.get("sample_id", 0)),
        )
        if (
            not prompt.get("prompt_hash")
            or row.get("prompt_hash") != prompt.get("prompt_hash")
            or row.get("generation_signature") != expected_signature
        ):
            return False
    evaluate_llm_predictions(config, "val")
    return True


def promote_selected_qwen_pilot(
    *,
    selected_variant: str,
    payload: dict[str, Any],
    selected_config: dict[str, Any],
    rows: list[dict[str, Any]],
    cell: Path,
) -> dict[str, Any]:
    """Seed the full validation output with exact compatible pilot successes.

    The selected pilot and full Qwen configuration use the same model,
    decoding parameters, train-only context and validation client profiles.
    ``run_explanation_generation`` independently recomputes and checks every
    generation signature before reusing these rows.
    """
    destination = Path(selected_config["output"]["paths_by_split"]["val"]["explanations"])
    rendered = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    if destination.exists() and destination.stat().st_size:
        if destination.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                "Refusing to overwrite an existing full validation explanation "
                f"checkpoint while promoting pilot rows: {destination}"
            )
    source_config = load_yaml(payload["pilot_configs"][selected_variant])
    source = Path(source_config["output"]["paths_by_split"]["val"]["explanations"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".pilot.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(destination)
    provenance = {
        "schema_version": 1,
        "status": "seeded_pending_signature_revalidation",
        "dataset": payload["dataset"],
        "run_id": payload["run_id"],
        "variant": selected_variant,
        "source": str(source),
        "source_sha256": file_sha256(source),
        "destination": str(destination),
        "destination_sha256": file_sha256(destination),
        "pilot_ids_hash": payload["pilot_ids_hash"],
        "n_rows": len(rows),
        "safety": (
            "The full generator must recompute prompt/generation signatures; "
            "incompatible rows are ignored and regenerated."
        ),
    }
    provenance["promotion_signature"] = fingerprint(provenance)
    atomic_write_json(cell / "qwen_pilot_promotion.json", provenance)
    return provenance


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    base = load_yaml(args.base_config)
    dataset = base["dataset"]["name"]
    manifest = load_benchmark_manifest(args.manifest, dataset)
    qwen = load_yaml(args.qwen_config)
    gpt = load_yaml(args.gpt_config)
    pilot_profile = load_yaml(args.pilot_config or args.qwen_config)
    cell = generated_cell(args.generated_root, args.run_id, dataset)
    cell.mkdir(parents=True, exist_ok=True)
    pilot_configs = {}
    for variant in VARIANTS:
        config = build_isolated_runtime_config(
            base,
            pilot_profile,
            manifest_path=args.manifest,
            manifest=manifest,
            run_id=args.run_id,
            variant=variant,
            results_root=args.results_root,
            mode="pilot",
        )
        path = cell / f"pilot_{variant}.yaml"
        write_runtime_config(path, config)
        audit(config, cell / "prompt_audits" / variant)
        pilot_configs[variant] = str(path)
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "dataset": dataset,
        "protocol": manifest["protocol"],
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_signature": manifest["manifest_signature"],
        "pilot_ids": manifest["roles"]["pilot_val"],
        "pilot_ids_hash": manifest["id_hashes"]["pilot_val"],
        "pilot_size": manifest["counts"]["pilot_val"],
        "base_config": str(args.base_config),
        "pilot_configs": pilot_configs,
        "pilot_config_sha256": {
            variant: file_sha256(path) for variant, path in pilot_configs.items()
        },
        "qwen_profile": qwen,
        "gpt_profile": gpt,
    }
    payload["materialization_signature"] = fingerprint(payload)
    atomic_write_json(cell / "pilot_materialization.json", payload)
    return payload


def run_pilot(payload: dict[str, Any], *, execute_api: bool, until_complete: bool) -> None:
    if not execute_api or not until_complete:
        raise ValueError("Pilot API execution requires --execute-api and --until-complete")
    expected_ids = {
        canonical_entity_id(value)
        for value in json.loads(
            Path(payload["pilot_ids"]).read_text(encoding="utf-8")
        )
    }
    for variant in VARIANTS:
        config_path = Path(payload["pilot_configs"][variant])
        if file_sha256(config_path) != payload["pilot_config_sha256"][variant]:
            raise RuntimeError(f"Pilot config changed after audit: {config_path}")
        config = load_yaml(config_path)
        if reuse_compatible_pilot_generation(config, expected_ids):
            print(
                f"[PILOT REUSE] {variant}: exact generation signatures "
                "verified; rebuilt metrics only"
            )
            continue
        subprocess.run([
            sys.executable, "run_pipeline.py", "--config", str(config_path),
            "--steps", "stats,prompts,cot,llm_eval", "--splits", "val",
            "--execute", "--until-complete",
        ], cwd=REPO_ROOT, check=True)


def select(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_benchmark_manifest(Path(payload["manifest"]), payload["dataset"])
    expected_ids = {
        canonical_entity_id(value)
        for value in json.loads(Path(payload["pilot_ids"]).read_text(encoding="utf-8"))
    }
    metrics, explanations, diagnostics = {}, {}, {}
    for variant, config_path in payload["pilot_configs"].items():
        config = load_yaml(config_path)
        root = Path(config["output"]["base_dir"])
        metrics[variant] = json.loads((root / "llm_metrics_val.json").read_text(encoding="utf-8"))
        explanations[variant] = exact_success_rows(root / "explanations_val.jsonl", expected_ids)
        diagnostics[variant] = rationale_diagnostics(
            explanations[variant], read_jsonl(root / "prompts_val.jsonl")
        )
    zero = "guided_zero_shot_v5"
    metric = "balanced_accuracy"
    paired = {
        variant: paired_primary_metric_delta(
            explanations[zero], explanations[variant], metric=metric, seed=137
        )
        for variant in VARIANTS if variant != zero
    }
    decision = select_prompt_variant_by_metric(metrics, paired, metric=metric)
    selected = decision["selected_variant"]
    base = load_yaml(payload["base_config"])
    cell = generated_cell(args.generated_root, payload["run_id"], payload["dataset"])
    selected_configs = {}
    built_configs = {}
    for name, profile in (("qwen", payload["qwen_profile"]), ("gpt_oss", payload["gpt_profile"])):
        config = build_isolated_runtime_config(
            base,
            profile,
            manifest_path=Path(payload["manifest"]),
            manifest=manifest,
            run_id=payload["run_id"],
            variant=selected,
            results_root=args.results_root,
            mode="full",
        )
        path = cell / f"selected_{name}.yaml"
        write_runtime_config(path, config)
        built_configs[name] = config
        selected_configs[name] = {"path": str(path), "sha256": file_sha256(path), "output_root": config["output"]["base_dir"]}
    promotion = promote_selected_qwen_pilot(
        selected_variant=selected,
        payload=payload,
        selected_config=built_configs["qwen"],
        rows=explanations[selected],
        cell=cell,
    )
    selection = {
        "schema_version": 1,
        "status": "completed",
        "run_id": payload["run_id"],
        "dataset": payload["dataset"],
        "protocol": payload["protocol"],
        "manifest_signature": payload["manifest_signature"],
        "pilot_ids_hash": payload["pilot_ids_hash"],
        "pilot_size": payload["pilot_size"],
        "direct_selection_metric": metric,
        "metrics": metrics,
        "paired_deltas": paired,
        "diagnostics": diagnostics,
        "qwen_pilot_reuse": promotion,
        **decision,
        "selected_configs": selected_configs,
        "full_expected_client_counts": {
            role: int(manifest["counts"][role]) for role in ("train", "val", "test")
        },
    }
    selection["selection_signature"] = fingerprint(selection)
    selection["selection_sha256"] = fingerprint(selection)
    atomic_write_json(cell / "prompt_selection.json", selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/isolated"))
    parser.add_argument("--generated-root", type=Path, default=Path("logs/runs/isolated/generated"))
    parser.add_argument("--qwen-config", type=Path, default=Path("configs/v2/qwen.yaml"))
    parser.add_argument("--gpt-config", type=Path, default=Path("configs/v2/gpt_oss.yaml"))
    parser.add_argument("--pilot-config", type=Path)
    parser.add_argument("--stage", choices=("materialize", "run", "select", "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "stage": args.stage,
        "manifest": str(args.manifest),
        "base_config": str(args.base_config),
        "variants": list(VARIANTS),
        "run_id": args.run_id,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    cell = generated_cell(
        args.generated_root, args.run_id,
        load_yaml(args.base_config)["dataset"]["name"],
    )
    materialization_path = cell / "pilot_materialization.json"
    if args.stage in {"materialize", "all"}:
        payload = materialize(args)
    else:
        payload = json.loads(materialization_path.read_text(encoding="utf-8"))
    if args.stage in {"run", "all"}:
        run_pilot(payload, execute_api=args.execute_api, until_complete=args.until_complete)
    if args.stage in {"select", "all"}:
        selection = select(payload, args)
        print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

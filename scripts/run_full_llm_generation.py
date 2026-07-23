#!/usr/bin/env python3
"""Run the complete reusable LLM artifact pass for one dataset/model cell.

The command is read-only unless all three execution guards are supplied.  A
completion marker is published only after every requested split has exactly one
successful explanation, direct prediction, and non-empty claims record for
every expected client.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import (  # noqa: E402
    atomic_write_json,
    file_sha256,
    files_fingerprint,
    fingerprint,
)
from src.experiments.config_builder import (  # noqa: E402
    EXPECTED_CLIENT_COUNTS,
    build_runtime_config,
    load_yaml,
    slug,
    write_runtime_config,
)


DATASETS = tuple(EXPECTED_CLIENT_COUNTS)
SPLITS = ("train", "val", "test")
LLM_STEPS = ("stats", "prompts", "cot", "llm_eval", "claims")


def parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [split for split in splits if split not in SPLITS]
    if not splits or unknown or len(splits) != len(set(splits)):
        raise ValueError(
            "--splits must contain unique values from train,val,test; "
            f"received {splits!r}"
        )
    return splits


def split_artifact_path(config: dict[str, Any], key: str, split: str) -> Path:
    configured = config.get("output", {}).get("paths_by_split", {}).get(split, {})
    artifact_aliases = {
        "clients_stats": "client_stats",
        "prompts": "prompts",
        "explanations": "explanations",
        "claims": "claims",
        "metrics": "direct_metrics",
    }
    if configured.get(artifact_aliases[key]):
        return Path(configured[artifact_aliases[key]])
    output_dir = Path(config["output"]["base_dir"])
    if key == "metrics":
        return output_dir / f"llm_metrics_{split}.json"
    base = Path(config["output"][key])
    return output_dir / f"{base.stem}_{split}{base.suffix or '.jsonl'}"


def pipeline_command(config_path: Path, splits: list[str]) -> list[str]:
    return [
        sys.executable,
        "run_pipeline.py",
        "--config",
        str(config_path),
        "--steps",
        ",".join(LLM_STEPS),
        "--splits",
        ",".join(splits),
        "--execute",
        "--until-complete",
    ]


def _selected_client_ids(config: dict[str, Any], split: str) -> set[int] | None:
    selection = config.get("dataset", {}).get("client_ids_by_split", {}).get(split)
    if selection is None:
        return None
    if isinstance(selection, (str, Path)):
        selection = json.loads(Path(selection).read_text(encoding="utf-8"))
    if not isinstance(selection, (list, tuple, set)):
        raise ValueError(f"Unsupported client filter for {split}: {selection!r}")
    return {int(value) for value in selection}


def expected_ids_by_split(
    config: dict[str, Any],
    splits: list[str],
) -> dict[str, set[int]]:
    """Read only customer columns and enforce the declared full-run counts."""
    dataset = config["dataset"]
    customer_column = dataset["columns"]["customer_id"]
    declared = dataset.get("expected_client_counts", {})
    result: dict[str, set[int]] = {}
    for split in splits:
        input_path = Path(dataset["splits"][split])
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing {split} input: {input_path}")
        identifiers: set[int] = set()
        for chunk in pd.read_csv(
            input_path,
            usecols=[customer_column],
            chunksize=250_000,
        ):
            identifiers.update(int(value) for value in chunk[customer_column].dropna())
        selected = _selected_client_ids(config, split)
        if selected is not None:
            missing_selection = selected - identifiers
            if missing_selection:
                raise ValueError(
                    f"Client filter for {split} contains {len(missing_selection)} "
                    "IDs absent from the input"
                )
            identifiers &= selected
        expected_count = declared.get(split)
        if expected_count is None:
            raise ValueError(f"No expected client count declared for split={split}")
        if len(identifiers) != int(expected_count):
            raise ValueError(
                f"Unexpected {split} client count: {len(identifiers)} != "
                f"{int(expected_count)}"
            )
        result[split] = identifiers
    return result


def validate_prompt_inputs(config: dict[str, Any]) -> None:
    prompt_config = config.get("prompts", {})
    base_dir = Path(prompt_config.get("base_dir", "."))
    for key in ("system", "user", "claims_system", "claims_user"):
        if not prompt_config.get(key):
            raise ValueError(f"Missing prompts.{key}")
        path = base_dir / prompt_config[key]
        if not path.is_file():
            raise FileNotFoundError(f"Missing prompt template: {path}")


def _records_by_customer(path: Path, expected_ids: set[int]) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required output: {path}")
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                customer_id = int(record["customer_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed record at {path}:{line_number}: {exc}") from exc
            if customer_id in records:
                raise ValueError(f"Duplicate customer_id={customer_id} in {path}")
            records[customer_id] = record
    observed = set(records)
    if observed != expected_ids:
        raise ValueError(
            f"Client coverage mismatch in {path}: "
            f"missing={len(expected_ids - observed)}, unexpected={len(observed - expected_ids)}"
        )
    return records


def _validate_split_outputs(
    config: dict[str, Any],
    split: str,
    expected_ids: set[int],
) -> dict[str, Any]:
    stats = _records_by_customer(
        split_artifact_path(config, "clients_stats", split), expected_ids
    )
    prompts = _records_by_customer(
        split_artifact_path(config, "prompts", split), expected_ids
    )
    explanations = _records_by_customer(
        split_artifact_path(config, "explanations", split), expected_ids
    )
    claims = _records_by_customer(
        split_artifact_path(config, "claims", split), expected_ids
    )
    del stats  # Coverage validation above is the required contract for stats.

    valid_labels = {int(value) for value in config["dataset"]["label_names"]}
    for customer_id, record in explanations.items():
        if record.get("error") or record.get("error_type"):
            raise ValueError(f"Failed explanation for {split} client {customer_id}")
        if not str(record.get("explanation", "")).strip():
            raise ValueError(f"Empty explanation for {split} client {customer_id}")
        try:
            prediction = int(record["predicted"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Unparseable prediction for {split} client {customer_id}"
            ) from exc
        if prediction not in valid_labels:
            raise ValueError(
                f"Out-of-domain prediction {prediction} for {split} client {customer_id}"
            )
        if not record.get("generation_signature"):
            raise ValueError(
                f"Missing generation signature for {split} client {customer_id}"
            )
        prompt_hash = prompts[customer_id].get("prompt_hash")
        if not prompt_hash or record.get("prompt_hash") != prompt_hash:
            raise ValueError(
                f"Prompt provenance mismatch for {split} client {customer_id}"
            )

    for customer_id, record in claims.items():
        if record.get("error") or record.get("error_type"):
            raise ValueError(f"Failed claims for {split} client {customer_id}")
        values = record.get("claims")
        if not isinstance(values, list) or not values or any(
            not str(value).strip() for value in values
        ):
            raise ValueError(f"Empty claims for {split} client {customer_id}")
        if not record.get("generation_signature") or not record.get(
            "source_explanation_hash"
        ):
            raise ValueError(
                f"Missing claims provenance for {split} client {customer_id}"
            )
        prompt_hash = explanations[customer_id].get("prompt_hash")
        if prompt_hash not in record.get("source_prompt_hashes", []):
            raise ValueError(
                f"Claims prompt provenance mismatch for {split} client {customer_id}"
            )

    metrics_path = split_artifact_path(config, "metrics", split)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing direct metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_count = len(expected_ids)
    required_metrics = {
        "split": split,
        "n_rows": expected_count,
        "n_scored": expected_count,
        "n_skipped": 0,
        "n_errors": 0,
    }
    for key, expected in required_metrics.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"Incomplete direct metrics in {metrics_path}: "
                f"{key}={metrics.get(key)!r}, expected {expected!r}"
            )
    if float(metrics.get("coverage", 0.0)) != 1.0:
        raise ValueError(f"Incomplete direct metric coverage in {metrics_path}")

    paths = [
        split_artifact_path(config, key, split)
        for key in ("clients_stats", "prompts", "explanations", "claims", "metrics")
    ]
    return {
        "expected_clients": expected_count,
        "artifacts": files_fingerprint(paths),
    }


def validate_completion(
    config: dict[str, Any],
    splits: list[str],
    expected_ids: dict[str, set[int]],
) -> dict[str, Any]:
    """Return immutable evidence only when every requested cell is complete."""
    split_evidence = {
        split: _validate_split_outputs(config, split, expected_ids[split])
        for split in splits
    }
    manifest_path = Path(config["output"]["base_dir"]) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = manifest.get("manifest_sha256")
    if not manifest_sha:
        raise ValueError(f"Manifest has no content identity: {manifest_path}")
    return {
        "manifest_sha256": manifest_sha,
        "manifest_file_sha256": file_sha256(manifest_path),
        "split_evidence": split_evidence,
    }


def completion_payload(
    config: dict[str, Any],
    splits: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    expected_counts = config["dataset"]["expected_client_counts"]
    identity = {
        "run_id": config["experiment"]["run_id"],
        "dataset": config["dataset"]["name"],
        "model_slug": config["experiment"]["model_slug"],
        "variant": config["experiment"]["variant"],
        "label_semantics": config["experiment"].get(
            "label_semantics", "standard"
        ),
        "splits": splits,
        "expected_counts": {split: int(expected_counts[split]) for split in splits},
        "manifest_sha256": evidence["manifest_sha256"],
        "split_evidence": evidence["split_evidence"],
    }
    return {
        "status": "completed",
        **identity,
        "completion_signature": fingerprint(identity),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--variant", default="robust_zero_shot_v2")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--run-id")
    parser.add_argument("--results-root", type=Path, default=Path("results/v2"))
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument(
        "--selected-config",
        type=Path,
        help=(
            "Use the content-addressed full config produced by run_prompt_pilot.py "
            "instead of constructing a variant directly."
        ),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="Content-addressed pilot selection artifact that owns --selected-config.",
    )
    parser.add_argument("--completion-marker", type=Path)
    parser.add_argument("--sampling-seed", type=int, default=137)
    parser.add_argument("--generation-seed", type=int, default=17)
    parser.add_argument("--claims-seed", type=int, default=17)
    parser.add_argument("--ml-seed", type=int, default=17)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    splits = parse_splits(args.splits)
    base_config_path = args.base_config or Path("configs") / f"{args.dataset}.yaml"
    base = load_yaml(base_config_path)
    if base.get("dataset", {}).get("name") != args.dataset:
        raise ValueError(
            f"Dataset mismatch: --dataset={args.dataset!r}, "
            f"config={base.get('dataset', {}).get('name')!r}"
        )
    profile = load_yaml(args.model_config)
    run_id = args.run_id or str(profile.get("experiment", {}).get("run_id", "reviewer-v2"))
    model_slug = slug(profile["experiment"]["model_slug"])
    if args.selected_config:
        if (
            args.runtime_config is not None
            and args.runtime_config.resolve() != args.selected_config.resolve()
        ):
            raise RuntimeError(
                "--runtime-config cannot differ from --selected-config"
            )
        if not args.selection or not args.selection.is_file():
            raise RuntimeError(
                "--selected-config requires an existing --selection artifact"
            )
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        claimed_selection_hash = selection.get("selection_sha256")
        unsigned_selection = {
            key: value
            for key, value in selection.items()
            if key != "selection_sha256"
        }
        if (
            selection.get("status") != "completed"
            or selection.get("run_id") != run_id
            or selection.get("dataset") != args.dataset
            or not claimed_selection_hash
            or fingerprint(unsigned_selection) != claimed_selection_hash
        ):
            raise RuntimeError(
                f"Stale or incompatible selection artifact: {args.selection}"
            )
        selected_entry = selection.get("selected_configs", {}).get(model_slug, {})
        if (
            Path(str(selected_entry.get("path", ""))).resolve()
            != args.selected_config.resolve()
            or selected_entry.get("sha256") != file_sha256(args.selected_config)
        ):
            raise RuntimeError(
                f"Selected config is not owned by {args.selection}: "
                f"{args.selected_config}"
            )
        config = load_yaml(args.selected_config)
        expected_identity = {
            "run_id": run_id,
            "dataset": args.dataset,
            "model_slug": model_slug,
        }
        observed_identity = {
            "run_id": config.get("experiment", {}).get("run_id"),
            "dataset": config.get("dataset", {}).get("name"),
            "model_slug": config.get("experiment", {}).get("model_slug"),
        }
        if observed_identity != expected_identity:
            raise RuntimeError(
                "Selected config identity mismatch: "
                f"observed={observed_identity}, expected={expected_identity}"
            )
        declared_counts = config.get("dataset", {}).get(
            "expected_client_counts", {}
        )
        if declared_counts != EXPECTED_CLIENT_COUNTS[args.dataset]:
            raise RuntimeError(
                f"Selected config client counts are incompatible: {declared_counts}"
            )
        selected_variant = str(config["experiment"]["variant"])
    else:
        config = build_runtime_config(
            base,
            profile,
            run_id=run_id,
            variant=args.variant,
            sampling_seed=args.sampling_seed,
            generation_seed=args.generation_seed,
            claims_seed=args.claims_seed,
            ml_seed=args.ml_seed,
            results_root=args.results_root,
            expected_client_counts=EXPECTED_CLIENT_COUNTS[args.dataset],
        )
        selected_variant = args.variant
    if int(config.get("pipeline", {}).get("n_explanation_samples", 1)) != 1:
        raise ValueError("Full LLM generation requires exactly one explanation per client")
    if int(config.get("pipeline", {}).get("n_claims_samples", 1)) != 1:
        raise ValueError("Full LLM generation requires exactly one claims response per client")

    generated_dir = Path("logs/runs") / run_id / "generated"
    runtime_config = args.runtime_config or args.selected_config or (
        generated_dir / f"{args.dataset}_{slug(selected_variant)}_{model_slug}.yaml"
    )
    completion_marker = args.completion_marker or (
        Path("logs/runs")
        / run_id
        / "completion"
        / f"{model_slug}_{args.dataset}.json"
    )
    config["output"]["completion_marker"] = str(completion_marker)
    command = pipeline_command(runtime_config, splits)
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "run_id": run_id,
        "dataset": args.dataset,
        "model_slug": model_slug,
        "variant": selected_variant,
        "label_semantics": config["experiment"].get(
            "label_semantics", "standard"
        ),
        "splits": splits,
        "steps": list(LLM_STEPS),
        "seeds": config["experiment"]["seeds"],
        "expected_counts": {
            split: config["dataset"]["expected_client_counts"][split]
            for split in splits
        },
        "input_paths": {
            split: config["dataset"]["input_paths_by_split"][split]
            for split in splits
        },
        "output_paths": {
            split: config["output"]["paths_by_split"][split]
            for split in splits
        },
        "runtime_config": str(runtime_config),
        "completion_marker": str(completion_marker),
        "selection": str(args.selection) if args.selection else None,
        "pipeline_command": command,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.execute:
        return
    if not args.execute_api or not args.until_complete:
        raise ValueError(
            "Execution requires --execute, --execute-api, and --until-complete"
        )

    validate_prompt_inputs(config)
    expected_ids = expected_ids_by_split(config, splits)
    # A failed rerun must not leave a stale success signal for the queue.
    completion_marker.unlink(missing_ok=True)
    if not args.selected_config:
        write_runtime_config(runtime_config, config)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    evidence = validate_completion(config, splits, expected_ids)
    marker = completion_payload(config, splits, evidence)
    atomic_write_json(completion_marker, marker)
    print(json.dumps({"completion_marker": str(completion_marker), **marker}, indent=2))


if __name__ == "__main__":
    main()

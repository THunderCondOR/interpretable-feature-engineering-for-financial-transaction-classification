"""Materialize and execute the neutral gender pilot without touching legacy outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import add_features, load_dataset
from src.experiments.artifacts import atomic_write_json, file_sha256, fingerprint, git_revision
from src.experiments.config_builder import (
    build_runtime_config,
    load_yaml,
    write_runtime_config,
)
from src.pipeline.llm_eval import balanced_accuracy_interval, summarize_prediction_rows

PILOT_VARIANTS = (
    "neutral_only",
    "neutral_robust_fewshot",
    "neutral_robust_zero_shot",
)
PILOT_SIZE = 400
PILOT_SAMPLING_SEED = 137
GENERATION_SEED = 17
FULL_SPLITS = ("train", "val", "test")
FULL_API_STEPS = "stats,prompts,cot,llm_eval,claims"


def stratified_pilot_ids(transactions, n_clients=400, seed=137):
    clients = transactions.groupby("customer_id", sort=False).agg(
        label=("label", "first"),
        transaction_count=("amount", "size"),
        transaction_volume=("amount", lambda values: values.abs().sum()),
    ).reset_index()
    for column in ("transaction_count", "transaction_volume"):
        clients[f"{column}_quartile"] = pd.qcut(
            clients[column].rank(method="first"),
            4,
            labels=False,
            duplicates="drop",
        )
    clients["stratum"] = clients[
        ["label", "transaction_count_quartile", "transaction_volume_quartile"]
    ].astype(str).agg("/".join, axis=1)
    target = min(n_clients, len(clients))
    counts = clients["stratum"].value_counts().sort_index()
    exact = counts / counts.sum() * target
    allocation = np.floor(exact).astype(int)
    remainder = target - int(allocation.sum())
    for stratum in (exact - allocation).sort_values(ascending=False).index[:remainder]:
        allocation[stratum] += 1
    rng, selected = np.random.default_rng(seed), []
    for stratum, group in clients.groupby("stratum", sort=True):
        take = min(int(allocation.get(stratum, 0)), len(group))
        selected.extend(
            rng.choice(group["customer_id"].to_numpy(), take, replace=False).tolist()
        )
    return sorted(int(value) for value in selected)


def build_plan(run_id: str, stage: str, model_config: Path) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": stage,
        "model_config": str(model_config),
        "dataset": "gender",
        "pilot_clients": PILOT_SIZE,
        "sampling_seed": PILOT_SAMPLING_SEED,
        "generation_seed": GENERATION_SEED,
        "pilot_variants": ["legacy", *PILOT_VARIANTS],
        "selection": {
            "split": "val",
            "primary": "balanced_accuracy",
            "equivalent_delta": 0.01,
            "prefer_when_equivalent": "neutral_robust_zero_shot",
        },
        "full_qwen": [*FULL_SPLITS, "direct", "claims"],
        "full_gpt_oss": [*FULL_SPLITS, "direct", "claims"],
    }


def _pipeline_command(config_path: Path, *, steps: str, splits: str) -> list[str]:
    return [
        sys.executable,
        "run_pipeline.py",
        "--config",
        str(config_path),
        "--steps",
        steps,
        "--splits",
        splits,
        "--execute",
        "--until-complete",
    ]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL artifact is missing: {path}")
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


def _rows_for_exact_clients(path: Path, client_ids: list[int]) -> list[dict[str, Any]]:
    """Return exactly one row per requested client, ignoring unrelated full-split rows."""
    expected = {int(value) for value in client_ids}
    if len(expected) != len(client_ids):
        raise ValueError("Expected client IDs contain duplicates")
    rows = [
        row for row in _read_jsonl(path)
        if int(row.get("customer_id", -1)) in expected
    ]
    observed = [int(row.get("customer_id", -1)) for row in rows]
    duplicates = sorted(
        value for value, count in Counter(observed).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate predictions for {len(duplicates)} requested clients in {path}"
        )
    missing = expected - set(observed)
    if missing:
        raise ValueError(
            f"Predictions missing {len(missing)} requested clients in {path}"
        )
    return sorted(rows, key=lambda row: int(row["customer_id"]))


def _completion_dir(run_id: str) -> Path:
    return Path("logs/runs") / run_id / "completion"


def _profile_generation_seed(profile: dict[str, Any]) -> int:
    """Generation seed is model configuration, never the pilot sampling seed."""
    return int(
        profile.get("generation", {}).get(
            "seed", profile.get("experiment", {}).get("seed", GENERATION_SEED)
        )
    )


def materialize(
    *,
    run_id: str,
    base_config_path: Path,
    model_config_path: Path,
    generated_dir: Path,
    results_root: Path,
) -> dict[str, Any]:
    base = load_yaml(base_config_path)
    profile = load_yaml(model_config_path)
    model_slug = profile["experiment"]["model_slug"]
    generated_dir.mkdir(parents=True, exist_ok=True)

    validation = add_features(load_dataset(base, "val"))
    pilot_ids = stratified_pilot_ids(
        validation, n_clients=PILOT_SIZE, seed=PILOT_SAMPLING_SEED
    )
    if len(pilot_ids) != PILOT_SIZE or len(set(pilot_ids)) != PILOT_SIZE:
        raise ValueError(
            f"Gender pilot requires exactly {PILOT_SIZE} unique validation clients; "
            f"selected {len(pilot_ids)} ({len(set(pilot_ids))} unique)"
        )
    pilot_ids_path = generated_dir / "gender_pilot_client_ids.json"
    atomic_write_json(pilot_ids_path, pilot_ids)

    # This is a launch preflight, not merely selection-time validation.  It must
    # fail before the first paid pilot request if the legacy comparison subset is
    # incomplete.  _metric deliberately reconstructs metrics from explanations
    # when the historical aggregate file is absent.
    legacy_metrics_path = Path(base["output"]["base_dir"]) / "llm_metrics_val.json"
    legacy_subset_metrics = _metric(legacy_metrics_path, pilot_ids)
    if int(legacy_subset_metrics.get("n_rows", -1)) != PILOT_SIZE or int(
        legacy_subset_metrics.get("n_scored", -1)
    ) != PILOT_SIZE:
        raise RuntimeError(
            "Legacy gender pilot preflight requires one successful parsed prediction "
            f"for every selected client; metrics={legacy_subset_metrics}"
        )

    configs: dict[str, str] = {}
    config_hashes: dict[str, str] = {}
    generation_seed = _profile_generation_seed(profile)
    for variant in PILOT_VARIANTS:
        config = build_runtime_config(
            base,
            profile,
            run_id=run_id,
            variant=variant,
            seed=generation_seed,
            results_root=results_root / "pilot",
            client_ids_by_split={"val": str(pilot_ids_path)},
        )
        config.setdefault("experiment", {})["sampling_seed"] = PILOT_SAMPLING_SEED
        config["experiment"]["generation_seed"] = generation_seed
        path = generated_dir / f"gender_{variant}_{model_slug}.yaml"
        write_runtime_config(path, config)
        configs[variant] = str(path)
        config_hashes[variant] = file_sha256(path)

    payload = {
        "run_id": run_id,
        "model_slug": model_slug,
        "pilot_ids": str(pilot_ids_path),
        "pilot_ids_sha256": file_sha256(pilot_ids_path),
        "pilot_size": PILOT_SIZE,
        "sampling_seed": PILOT_SAMPLING_SEED,
        "generation_seed": generation_seed,
        "pilot_configs": configs,
        "pilot_config_sha256": config_hashes,
        "legacy_metrics": str(legacy_metrics_path),
        "legacy_explanations": str(legacy_metrics_path.parent / "explanations_val.jsonl"),
        "legacy_subset_metrics": legacy_subset_metrics,
        "legacy_subset_metrics_sha256": fingerprint(legacy_subset_metrics),
    }
    payload["materialization_sha256"] = fingerprint(payload)
    atomic_write_json(generated_dir / f"gender_{model_slug}.materialized.json", payload)
    return payload


def _metric(path: Path, client_ids: list[int] | None = None) -> dict[str, Any]:
    explanations = path.parent / "explanations_val.jsonl"
    rows: list[dict[str, Any]] | None = None
    if client_ids is not None:
        rows = _rows_for_exact_clients(explanations, client_ids)
        payload = summarize_prediction_rows(rows, split="val")
    elif path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif explanations.exists():
        rows = _read_jsonl(explanations)
        payload = summarize_prediction_rows(rows, split="val")
    else:
        raise FileNotFoundError(
            f"Missing validation metrics and per-client predictions: {path}, {explanations}"
        )
    if "balanced_accuracy" not in payload:
        raise ValueError(f"Validation metrics lack balanced_accuracy: {path}")
    if "balanced_accuracy_ci" not in payload and explanations.exists():
        if rows is None:
            rows = _read_jsonl(explanations)
        scored = [
            (int(row["label"]), int(row["predicted"]))
            for row in rows
            if not row.get("error")
            and int(row.get("label", -1)) >= 0
            and row.get("predicted") is not None
        ]
        if scored:
            labels, predictions = zip(*scored)
            payload["balanced_accuracy_ci"] = balanced_accuracy_interval(
                labels,
                predictions,
                n_bootstrap=1000,
                seed=PILOT_SAMPLING_SEED,
            )
    return payload


def _intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_ci = left.get("balanced_accuracy_ci")
    right_ci = right.get("balanced_accuracy_ci")
    if not left_ci or not right_ci:
        return False
    return max(float(left_ci["lower"]), float(right_ci["lower"])) <= min(
        float(left_ci["upper"]), float(right_ci["upper"])
    )


def _expected_client_ids(config: dict[str, Any], split: str) -> list[int]:
    frame = load_dataset(config, split)
    ids = [int(value) for value in frame["customer_id"].drop_duplicates().tolist()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Dataset split {split} contains duplicate client identities")
    return sorted(ids)


def _validate_explanations(
    config: dict[str, Any], split: str, expected_ids: list[int]
) -> None:
    out_dir = Path(config["output"]["base_dir"])
    rows = _rows_for_exact_clients(out_dir / f"explanations_{split}.jsonl", expected_ids)
    invalid = [
        int(row["customer_id"])
        for row in rows
        if row.get("error")
        or row.get("predicted") is None
        or not str(row.get("explanation", "")).strip()
    ]
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} incomplete explanation records for split={split}: "
            f"{invalid[:10]}"
        )
    metrics_path = out_dir / f"llm_metrics_{split}.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing direct LLM metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if int(metrics.get("n_rows", -1)) != len(expected_ids):
        raise RuntimeError(
            f"Metrics row count mismatch for split={split}: "
            f"{metrics.get('n_rows')} != {len(expected_ids)}"
        )
    if int(metrics.get("n_scored", -1)) != len(expected_ids):
        raise RuntimeError(
            f"Metrics are not complete for split={split}: "
            f"n_scored={metrics.get('n_scored')} expected={len(expected_ids)}"
        )


def _validate_claims(config: dict[str, Any], split: str, expected_ids: list[int]) -> None:
    path = Path(config["output"]["base_dir"]) / f"claims_{split}.jsonl"
    rows = _rows_for_exact_clients(path, expected_ids)
    invalid = [
        int(row["customer_id"])
        for row in rows
        if row.get("error") or not row.get("claims")
    ]
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} incomplete claim records for split={split}: {invalid[:10]}"
        )


def validate_complete_generation(
    config: dict[str, Any],
    *,
    splits: tuple[str, ...] = FULL_SPLITS,
    require_claims: bool = True,
    expected_ids_by_split: dict[str, list[int]] | None = None,
) -> tuple[dict[str, int], str]:
    """Validate exact client-level completeness and return counts + manifest hash."""
    counts: dict[str, int] = {}
    for split in splits:
        expected_ids = (
            expected_ids_by_split[split]
            if expected_ids_by_split is not None
            else _expected_client_ids(config, split)
        )
        _validate_explanations(config, split, expected_ids)
        if require_claims:
            _validate_claims(config, split, expected_ids)
        counts[split] = len(expected_ids)

    manifest_path = Path(config["output"]["base_dir"]) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = str(manifest.get("manifest_sha256", ""))
    if not manifest_hash:
        raise RuntimeError(f"Manifest has no manifest_sha256: {manifest_path}")
    return counts, manifest_hash


def _write_pilot_completion(
    materialized: dict[str, Any]
) -> Path:
    ids = json.loads(Path(materialized["pilot_ids"]).read_text(encoding="utf-8"))
    manifest_hashes: dict[str, str] = {}
    for variant, config_path in materialized["pilot_configs"].items():
        if file_sha256(config_path) != materialized["pilot_config_sha256"][variant]:
            raise RuntimeError(f"Pilot config changed after materialization: {config_path}")
        config = load_yaml(config_path)
        _counts, manifest_hash = validate_complete_generation(
            config,
            splits=("val",),
            require_claims=False,
            expected_ids_by_split={"val": ids},
        )
        manifest_hashes[variant] = manifest_hash
    marker = {
        "run_id": materialized["run_id"],
        "dataset": "gender",
        "model_slug": materialized["model_slug"],
        "selected_variant": "pilot",
        "variants": list(PILOT_VARIANTS),
        "splits": ["val"],
        "expected_counts": {"val": PILOT_SIZE},
        "pilot_ids_sha256": materialized["pilot_ids_sha256"],
        "pilot_config_sha256": materialized["pilot_config_sha256"],
        "manifest_sha256": manifest_hashes,
        "status": "completed",
    }
    path = _completion_dir(materialized["run_id"]) / "qwen_gender_pilot.json"
    atomic_write_json(path, marker)
    return path


def _validate_pilot_completion(materialized: dict[str, Any]) -> dict[str, Any]:
    path = _completion_dir(materialized["run_id"]) / "qwen_gender_pilot.json"
    if not path.is_file():
        raise FileNotFoundError(f"Completed gender pilot is required for selection: {path}")
    marker = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "run_id": materialized["run_id"],
        "dataset": "gender",
        "model_slug": materialized["model_slug"],
        "status": "completed",
        "pilot_ids_sha256": materialized["pilot_ids_sha256"],
        "pilot_config_sha256": materialized["pilot_config_sha256"],
    }
    mismatched = [key for key, value in expected.items() if marker.get(key) != value]
    if mismatched:
        raise RuntimeError(f"Stale or incompatible gender pilot marker ({mismatched}): {path}")
    ids = json.loads(Path(materialized["pilot_ids"]).read_text(encoding="utf-8"))
    for variant, config_path in materialized["pilot_configs"].items():
        _counts, current_manifest = validate_complete_generation(
            load_yaml(config_path),
            splits=("val",),
            require_claims=False,
            expected_ids_by_split={"val": ids},
        )
        if marker.get("manifest_sha256", {}).get(variant) != current_manifest:
            raise RuntimeError(
                f"Pilot output manifest changed after completion for variant={variant}"
            )
    return marker


def _selected_config_entry(selection: dict[str, Any], model_slug: str) -> dict[str, str]:
    entry = selection.get("selected_configs", {}).get(model_slug)
    if not isinstance(entry, dict):
        raise RuntimeError(f"Selection has no config for model_slug={model_slug!r}")
    path = Path(str(entry.get("path", "")))
    expected_hash = str(entry.get("sha256", ""))
    if not path.is_file() or not expected_hash:
        raise RuntimeError(f"Selected config is missing for model_slug={model_slug!r}")
    if file_sha256(path) != expected_hash:
        raise RuntimeError(f"Selected gender config changed: {path}")
    return {"path": str(path), "sha256": expected_hash}


def _write_full_completion(
    *,
    run_id: str,
    selection: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
) -> Path:
    model_slug = str(config["experiment"]["model_slug"])
    expected_counts, manifest_sha256 = validate_complete_generation(config)
    marker = {
        "run_id": run_id,
        "dataset": "gender",
        "model_slug": model_slug,
        "selected_variant": selection["selected_variant"],
        "splits": list(FULL_SPLITS),
        "expected_counts": expected_counts,
        "manifest_sha256": manifest_sha256,
        "selected_config": str(config_path),
        "selected_config_sha256": file_sha256(config_path),
        "selection_sha256": selection["selection_sha256"],
        "status": "completed",
    }
    path = _completion_dir(run_id) / f"{model_slug}_gender.json"
    atomic_write_json(path, marker)
    return path


def select_variant(
    materialized: dict[str, Any],
    *,
    base_config_path: Path,
    model_config_path: Path,
    gpt_model_config_path: Path,
    generated_dir: Path,
    results_root: Path,
) -> dict[str, Any]:
    _validate_pilot_completion(materialized)
    pilot_ids = json.loads(Path(materialized["pilot_ids"]).read_text(encoding="utf-8"))
    if file_sha256(materialized["pilot_ids"]) != materialized["pilot_ids_sha256"]:
        raise RuntimeError("Gender pilot client IDs changed after materialization")
    if fingerprint(materialized["legacy_subset_metrics"]) != materialized[
        "legacy_subset_metrics_sha256"
    ]:
        raise RuntimeError("Legacy gender subset metrics changed after materialization")
    candidates: dict[str, dict[str, Any]] = {
        "legacy": dict(materialized["legacy_subset_metrics"])
    }
    for variant, config_path in materialized["pilot_configs"].items():
        if file_sha256(config_path) != materialized["pilot_config_sha256"][variant]:
            raise RuntimeError(f"Pilot config changed after execution: {config_path}")
        config = load_yaml(config_path)
        candidates[variant] = _metric(
            Path(config["output"]["base_dir"]) / "llm_metrics_val.json",
            pilot_ids,
        )

    ranked = sorted(
        PILOT_VARIANTS,
        key=lambda name: (-float(candidates[name]["balanced_accuracy"]), name),
    )
    best = ranked[0]
    preferred = "neutral_robust_zero_shot"
    if (
        preferred in candidates
        and float(candidates[best]["balanced_accuracy"])
        - float(candidates[preferred]["balanced_accuracy"])
        <= 0.01
        and _intervals_overlap(candidates[best], candidates[preferred])
    ):
        best = preferred
    base = load_yaml(base_config_path)
    selected_configs: dict[str, dict[str, str]] = {}
    for profile_path in (model_config_path, gpt_model_config_path):
        profile = load_yaml(profile_path)
        model_slug = str(profile["experiment"]["model_slug"])
        generation_seed = _profile_generation_seed(profile)
        full_config = build_runtime_config(
            base,
            profile,
            run_id=materialized["run_id"],
            variant=best,
            seed=generation_seed,
            results_root=results_root,
        )
        full_config.setdefault("experiment", {})["sampling_seed"] = PILOT_SAMPLING_SEED
        full_config["experiment"]["generation_seed"] = generation_seed
        selected_config_path = generated_dir / f"gender_selected_{model_slug}.yaml"
        write_runtime_config(selected_config_path, full_config)
        selected_configs[model_slug] = {
            "path": str(selected_config_path),
            "sha256": file_sha256(selected_config_path),
            "model_config": str(profile_path),
            "model_config_sha256": file_sha256(profile_path),
        }

    qwen_slug = str(load_yaml(model_config_path)["experiment"]["model_slug"])
    selection = {
        "run_id": materialized["run_id"],
        "selected_variant": best,
        "selection_split": "val",
        "pilot_ids": materialized["pilot_ids"],
        "pilot_ids_sha256": materialized["pilot_ids_sha256"],
        "pilot_config_sha256": materialized["pilot_config_sha256"],
        "metrics": candidates,
        "metrics_sha256": fingerprint(candidates),
        "selected_configs": selected_configs,
        "git_revision": git_revision(REPO_ROOT),
        # Compatibility aliases for callers written before multi-model selection.
        "selected_config": selected_configs[qwen_slug]["path"],
        "selected_config_sha256": selected_configs[qwen_slug]["sha256"],
    }
    selection["selection_sha256"] = fingerprint(selection)
    atomic_write_json(generated_dir / "gender_selection.json", selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v2")
    parser.add_argument("--stage", choices=["plan", "pilot", "select", "full", "gpt"], default="plan")
    parser.add_argument("--base-config", type=Path, default=Path("configs/gender.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/v2/qwen.yaml"))
    parser.add_argument(
        "--gpt-model-config", type=Path, default=Path("configs/v2/gpt_oss.yaml")
    )
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/v2"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()

    generated_dir = args.generated_dir or Path("logs/runs") / args.run_id / "generated"
    plan = build_plan(args.run_id, args.stage, args.model_config)
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", **plan}, indent=2))
    if not args.execute:
        return
    if args.execute_api and not args.until_complete:
        raise ValueError("--execute-api requires --until-complete")

    materialized = None
    if args.stage in {"plan", "pilot", "select"}:
        materialized = materialize(
            run_id=args.run_id,
            base_config_path=args.base_config,
            model_config_path=args.model_config,
            generated_dir=generated_dir,
            results_root=args.results_root,
        )
    if args.stage == "plan":
        return
    if args.stage == "pilot":
        if not args.execute_api:
            print("Pilot configs materialized; API execution was not requested.")
            return
        for config_path in materialized["pilot_configs"].values():
            _run(_pipeline_command(
                Path(config_path),
                steps="stats,prompts,cot,llm_eval",
                splits="val",
            ))
        marker_path = _write_pilot_completion(materialized)
        print(f"Gender pilot completion -> {marker_path}")
        return

    if args.stage == "select":
        selection = select_variant(
            materialized,
            base_config_path=args.base_config,
            model_config_path=args.model_config,
            gpt_model_config_path=args.gpt_model_config,
            generated_dir=generated_dir,
            results_root=args.results_root,
        )
        print(json.dumps(selection, indent=2, ensure_ascii=False))
        return

    selection_path = generated_dir / "gender_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Selection artifact is required before {args.stage}: {selection_path}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("run_id") != args.run_id:
        raise RuntimeError(
            f"Stale gender selection for run {selection.get('run_id')!r}; "
            f"expected {args.run_id!r}"
        )
    claimed_selection_hash = selection.get("selection_sha256")
    unsigned_selection = {
        key: value for key, value in selection.items() if key != "selection_sha256"
    }
    if not claimed_selection_hash or fingerprint(unsigned_selection) != claimed_selection_hash:
        raise RuntimeError("Gender selection artifact is not content-compatible")
    profile = load_yaml(args.model_config)
    model_slug = str(profile["experiment"]["model_slug"])
    selected_entry = _selected_config_entry(selection, model_slug)
    selected_config = Path(selected_entry["path"])
    if args.stage == "full":
        if not args.execute_api:
            print("Full Qwen config is ready; API execution was not requested.")
            return
        _run(_pipeline_command(
            selected_config,
            steps=FULL_API_STEPS,
            splits=",".join(FULL_SPLITS),
        ))
        marker_path = _write_full_completion(
            run_id=args.run_id,
            selection=selection,
            config=load_yaml(selected_config),
            config_path=selected_config,
        )
        print(f"Gender full completion -> {marker_path}")
        return

    if not args.execute_api:
        print("GPT config is ready; API execution was not requested.")
        return
    _run(_pipeline_command(
        selected_config,
        steps=FULL_API_STEPS,
        splits=",".join(FULL_SPLITS),
    ))
    marker_path = _write_full_completion(
        run_id=args.run_id,
        selection=selection,
        config=load_yaml(selected_config),
        config_path=selected_config,
    )
    print(f"Gender GPT completion -> {marker_path}")


if __name__ == "__main__":
    main()

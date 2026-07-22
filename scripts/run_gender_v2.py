"""Materialize and execute the neutral gender pilot without touching legacy outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import add_features, load_dataset
from src.experiments.artifacts import atomic_write_json, file_sha256
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
        "pilot_clients": 400,
        "sampling_seed": 137,
        "pilot_variants": ["legacy", *PILOT_VARIANTS],
        "selection": {
            "split": "val",
            "primary": "balanced_accuracy",
            "equivalent_delta": 0.01,
            "prefer_when_equivalent": "neutral_robust_zero_shot",
        },
        "full_qwen": ["train", "val", "test", "direct", "claims", "clusters", "cot_ml", "concat_ml"],
        "full_gpt_oss": ["test direct", "grounding claims"],
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
    pilot_ids = stratified_pilot_ids(validation, n_clients=400, seed=137)
    pilot_ids_path = generated_dir / "gender_pilot_client_ids.json"
    atomic_write_json(pilot_ids_path, pilot_ids)

    configs: dict[str, str] = {}
    for variant in PILOT_VARIANTS:
        config = build_runtime_config(
            base,
            profile,
            run_id=run_id,
            variant=variant,
            seed=137,
            results_root=results_root / "pilot",
            client_ids_by_split={"val": str(pilot_ids_path)},
        )
        path = generated_dir / f"gender_{variant}_{model_slug}.yaml"
        write_runtime_config(path, config)
        configs[variant] = str(path)

    payload = {
        "run_id": run_id,
        "model_slug": model_slug,
        "pilot_ids": str(pilot_ids_path),
        "pilot_configs": configs,
        "legacy_metrics": str(Path(base["output"]["base_dir"]) / "llm_metrics_val.json"),
    }
    atomic_write_json(generated_dir / f"gender_{model_slug}.materialized.json", payload)
    return payload


def _metric(path: Path, client_ids: list[int] | None = None) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing validation metrics: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    explanations = path.parent / "explanations_val.jsonl"
    if client_ids is not None:
        if not explanations.exists():
            raise FileNotFoundError(
                f"Per-client predictions are required for pilot comparison: {explanations}"
            )
        expected = {int(value) for value in client_ids}
        rows = [
            json.loads(line)
            for line in explanations.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [row for row in rows if int(row.get("customer_id", -1)) in expected]
        observed = [int(row.get("customer_id", -1)) for row in rows]
        if len(observed) != len(set(observed)):
            raise ValueError(f"Duplicate pilot predictions in {explanations}")
        missing = expected - set(observed)
        if missing:
            raise ValueError(
                f"Pilot predictions missing {len(missing)} expected clients in {explanations}"
            )
        payload = summarize_prediction_rows(rows, split="val")
    if "balanced_accuracy" not in payload:
        raise ValueError(f"Validation metrics lack balanced_accuracy: {path}")
    if "balanced_accuracy_ci" not in payload and explanations.exists():
        rows = [
            json.loads(line)
            for line in explanations.read_text(encoding="utf-8").splitlines()
            if line.strip()
            and (client_ids is None or int(json.loads(line).get("customer_id", -1)) in expected)
        ]
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
                labels, predictions, n_bootstrap=1000, seed=137
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


def select_variant(
    materialized: dict[str, Any],
    *,
    base_config_path: Path,
    model_config_path: Path,
    generated_dir: Path,
    results_root: Path,
) -> dict[str, Any]:
    pilot_ids = json.loads(Path(materialized["pilot_ids"]).read_text(encoding="utf-8"))
    candidates: dict[str, dict[str, Any]] = {
        "legacy": _metric(Path(materialized["legacy_metrics"]), pilot_ids)
    }
    for variant, config_path in materialized["pilot_configs"].items():
        config = load_yaml(config_path)
        candidates[variant] = _metric(
            Path(config["output"]["base_dir"]) / "llm_metrics_val.json",
            pilot_ids,
        )

    ranked = sorted(
        candidates,
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
    if best == "legacy":
        raise RuntimeError(
            "Legacy won validation selection; no neutral full run is authorized. "
            "Inspect pilot metrics and grounding before proceeding."
        )

    base = load_yaml(base_config_path)
    profile = load_yaml(model_config_path)
    full_config = build_runtime_config(
        base,
        profile,
        run_id=materialized["run_id"],
        variant=best,
        seed=int(profile.get("experiment", {}).get("seed", 17)),
        results_root=results_root,
    )
    selected_config_path = generated_dir / (
        f"gender_selected_{profile['experiment']['model_slug']}.yaml"
    )
    write_runtime_config(selected_config_path, full_config)
    selection = {
        "run_id": materialized["run_id"],
        "selected_variant": best,
        "selection_split": "val",
        "metrics": candidates,
        "selected_config": str(selected_config_path),
        "selected_config_sha256": file_sha256(selected_config_path),
    }
    atomic_write_json(generated_dir / "gender_selection.json", selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v2")
    parser.add_argument("--stage", choices=["plan", "pilot", "select", "full", "gpt"], default="plan")
    parser.add_argument("--base-config", type=Path, default=Path("configs/gender.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/v2/qwen.yaml"))
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
        return

    if args.stage == "select":
        selection = select_variant(
            materialized,
            base_config_path=args.base_config,
            model_config_path=args.model_config,
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
    selected_config = Path(str(selection.get("selected_config", "")))
    expected_config_hash = selection.get("selected_config_sha256")
    if not selected_config.is_file() or not expected_config_hash:
        raise RuntimeError("Gender selection lacks a content-addressed selected config")
    if file_sha256(selected_config) != expected_config_hash:
        raise RuntimeError(f"Selected gender config changed: {selected_config}")
    if args.stage == "full":
        if not args.execute_api:
            print("Full Qwen config is ready; API execution was not requested.")
            return
        _run(_pipeline_command(
            Path(selection["selected_config"]),
            steps="stats,prompts,cot,llm_eval,claims,cot_features,ml",
            splits="train,val,test",
        ))
        return

    # GPT uses the Qwen-selected prompt variant but an independent model profile,
    # limiter state, output root, and generation signature.
    base = load_yaml(args.base_config)
    profile = load_yaml(args.model_config)
    gpt_config = build_runtime_config(
        base,
        profile,
        run_id=args.run_id,
        variant=selection["selected_variant"],
        seed=int(profile.get("experiment", {}).get("seed", 17)),
        results_root=args.results_root,
    )
    gpt_config_path = generated_dir / "gender_selected_gpt_oss.yaml"
    write_runtime_config(gpt_config_path, gpt_config)
    if not args.execute_api:
        print("GPT config is ready; API execution was not requested.")
        return
    _run(_pipeline_command(
        gpt_config_path,
        steps="stats,prompts,cot,llm_eval,claims",
        splits="test",
    ))


if __name__ == "__main__":
    main()

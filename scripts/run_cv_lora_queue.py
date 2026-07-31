#!/usr/bin/env python3
"""Sequential Qwen LoRA queue on immutable benchmark fold manifests.

Dry-run is the default.  The 8B model is the default execution target; 32B
must be requested explicitly so it cannot accidentally multiply a long run.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.benchmark_registry import BENCHMARKS
from src.experiments.artifacts import atomic_write_json, fingerprint
from src.experiments.config_builder import load_yaml
from src.models.lora_trainer import train


SUPPORTED_MODELS = {"qwen3_8b", "qwen3_32b"}


def materialize_config(
    *,
    dataset: str,
    fold: int,
    prepared_root: Path,
    output_root: Path,
    lora_profile: dict,
    protocol: str | None = None,
) -> tuple[dict, dict]:
    spec = BENCHMARKS[dataset]
    protocol = protocol or spec.protocol
    root = prepared_root / dataset / protocol
    benchmark_path = root / "benchmark_manifest.json"
    fold_path = root / f"fold_{fold}" / "fold_manifest.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    manifest = json.loads(fold_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != protocol:
        raise ValueError(f"Wrong fold protocol: {fold_path}")

    ids = manifest["ids"]
    train_ids = set(ids["inner_train"])
    validation_ids = set(ids["inner_validation"])
    test_ids = set(ids["outer_test"])
    if train_ids & validation_ids or (train_ids | validation_ids) & test_ids:
        raise ValueError(f"LoRA split leakage in {fold_path}")
    if train_ids | validation_ids != set(ids["outer_train"]):
        raise ValueError(f"Inner split does not cover outer train in {fold_path}")

    config = copy.deepcopy(load_yaml(f"configs/v5/{dataset}.yaml"))
    config["dataset"]["benchmark_protocol"] = protocol
    events = str(Path(benchmark["events"]))
    fold_root = fold_path.parent
    config["dataset"]["splits"] = {
        "train": events,
        "val": events,
        "test": events,
    }
    config["dataset"]["client_ids_by_split"] = {
        "train": str(fold_root / "inner_train_ids.json"),
        "val": str(fold_root / "inner_validation_ids.json"),
        "test": str(fold_root / "outer_test_ids.json"),
    }
    run_root = output_root / dataset / protocol / f"fold_{fold}"
    config["output"]["base_dir"] = str(run_root)
    config["lora"] = copy.deepcopy(lora_profile)
    config["lora_provenance"] = {
        "dataset": dataset,
        "protocol": protocol,
        "fold": fold,
        "fold_signature": manifest["fold_signature"],
        "split_roles": {
            "train": "inner_train",
            "val": "inner_validation",
            "test": "outer_test",
        },
        "counts": {
            "train": len(train_ids),
            "val": len(validation_ids),
            "test": len(test_ids),
        },
    }
    return config, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="datafusion_education")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--models", default="qwen3_8b")
    parser.add_argument(
        "--protocol-map",
        default=(
            "datafusion_education=mbd_5fold_seed42,"
            "berka=unittab_70_30_5seed"
        ),
        help="Comma-separated dataset=prepared_protocol mapping.",
    )
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("data/benchmarks_v5")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/v5/lora")
    )
    parser.add_argument(
        "--lora-config", type=Path, default=Path("configs/v5/lora_qwen.yaml")
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    datasets = tuple(value for value in args.datasets.split(",") if value)
    folds = tuple(int(value) for value in args.folds.split(",") if value)
    models = tuple(value for value in args.models.split(",") if value)
    protocol_map = {}
    for item in args.protocol_map.split(","):
        dataset, separator, protocol = item.partition("=")
        if not separator or not dataset or not protocol:
            raise ValueError(f"Invalid protocol mapping: {item!r}")
        protocol_map[dataset] = protocol
    unknown = set(datasets) - set(BENCHMARKS)
    unknown_models = set(models) - SUPPORTED_MODELS
    if unknown or unknown_models:
        raise ValueError(
            f"Unsupported datasets/models: datasets={sorted(unknown)}, "
            f"models={sorted(unknown_models)}"
        )
    profile = load_yaml(args.lora_config)
    configured_models = {item["name"] for item in profile["models"]}
    if not set(models) <= configured_models:
        raise ValueError("Requested model is absent from the LoRA profile")

    jobs = []
    materialized = []
    # Complete every 8B fold before beginning the optional 32B queue.
    for model in models:
        for dataset in datasets:
            protocol = protocol_map.get(dataset, BENCHMARKS[dataset].protocol)
            for fold in folds:
                config, manifest = materialize_config(
                    dataset=dataset,
                    fold=fold,
                    prepared_root=args.prepared_root,
                    output_root=args.output_root,
                    lora_profile=profile,
                    protocol=protocol,
                )
                signature = fingerprint({
                    "config": config,
                    "fold_signature": manifest["fold_signature"],
                    "model": model,
                })
                jobs.append({
                    "dataset": dataset,
                    "protocol": config["dataset"]["benchmark_protocol"],
                    "fold": fold,
                    "models": [model],
                    "counts": config["lora_provenance"]["counts"],
                    "signature": signature,
                    "output": config["output"]["base_dir"],
                })
                materialized.append((config, signature, model))

    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "execution": "sequential",
        "jobs": jobs,
        "note": "qwen3_32b runs only when explicitly requested",
    }, indent=2))
    if not args.execute:
        return

    for config, signature, model in materialized:
        run_root = Path(config["output"]["base_dir"])
        completion = run_root / f"completion_{model}.json"
        run_manifest = run_root / f"run_manifest_{model}.json"
        if completion.is_file():
            payload = json.loads(completion.read_text(encoding="utf-8"))
            if payload.get("signature") == signature and payload.get("status") == "completed":
                print(f"Reuse completed LoRA fold -> {run_root}")
                continue
            raise RuntimeError(f"Incompatible existing LoRA output: {run_root}")
        if run_manifest.is_file():
            payload = json.loads(run_manifest.read_text(encoding="utf-8"))
            if payload.get("signature") != signature:
                raise RuntimeError(
                    f"Incompatible incomplete LoRA output: {run_root}"
                )
        run_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_manifest, {
            "status": "running",
            "signature": signature,
            **config["lora_provenance"],
            "models": [model],
        })
        train(config, model_names=[model])
        atomic_write_json(completion, {
            "status": "completed",
            "signature": signature,
            **config["lora_provenance"],
            "models": [model],
        })


if __name__ == "__main__":
    main()

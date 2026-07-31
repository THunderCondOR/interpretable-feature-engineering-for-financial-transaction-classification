#!/usr/bin/env python3
"""Sequential Qwen3 LoRA queue for Gender, Rosbank and Age v4 splits."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json, file_sha256, fingerprint
from src.experiments.config_builder import load_yaml
from src.models.lora_trainer import train


DATASETS = ("rosbank", "gender", "age")


def materialize(
    dataset: str, output_root: Path, profile: dict
) -> tuple[dict, str]:
    config = copy.deepcopy(load_yaml(Path("configs") / f"{dataset}.yaml"))
    config["output"]["base_dir"] = str(output_root / dataset)
    config["lora"] = copy.deepcopy(profile)
    split_hashes = {
        split: file_sha256(Path(config["dataset"]["splits"][split]))
        for split in ("train", "val", "test")
    }
    provenance = {
        "dataset": dataset,
        "split_hashes": split_hashes,
        "split_roles": {"train": "train", "val": "validation", "test": "test"},
    }
    config["lora_provenance"] = provenance
    return config, fingerprint({"config": config, "provenance": provenance})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--models", default="qwen3_8b")
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("results/v5/lora/reviewer-v10-all-datasets/original"),
    )
    parser.add_argument(
        "--lora-config", type=Path, default=Path("configs/v5/lora_qwen.yaml")
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    datasets = tuple(value for value in args.datasets.split(",") if value)
    unknown = set(datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unsupported original datasets: {sorted(unknown)}")
    models = [value for value in args.models.split(",") if value]
    profile = load_yaml(args.lora_config)
    configured = {row["name"] for row in profile["models"]}
    if not set(models) <= configured:
        raise ValueError(f"Unknown LoRA models: {sorted(set(models)-configured)}")
    jobs = []
    configs = []
    for model in models:
        for dataset in datasets:
            config, signature = materialize(dataset, args.output_root, profile)
            jobs.append({
                "dataset": dataset, "model": model,
                "output": config["output"]["base_dir"], "signature": signature,
            })
            configs.append((config, signature, model))
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "execution": "sequential", "jobs": jobs,
    }, indent=2))
    if not args.execute:
        return
    for config, signature, model in configs:
        root = Path(config["output"]["base_dir"])
        marker = root / f"completion_{model}.json"
        manifest = root / f"run_manifest_{model}.json"
        if marker.is_file():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("status") == "completed" and payload.get("signature") == signature:
                print(f"Reuse completed original LoRA -> {root}")
                continue
            raise RuntimeError(f"Incompatible original LoRA completion: {root}")
        if manifest.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("signature") != signature:
                raise RuntimeError(f"Incompatible original LoRA manifest: {root}")
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest, {
            "status": "running", "signature": signature,
            **config["lora_provenance"], "model": model,
        })
        train(config, model_names=[model])
        atomic_write_json(marker, {
            "status": "completed", "signature": signature,
            **config["lora_provenance"], "model": model,
        })


if __name__ == "__main__":
    main()

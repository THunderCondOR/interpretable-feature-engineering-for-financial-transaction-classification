#!/usr/bin/env python3
"""Configure a deterministic Age full-run subset without making API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.client_sampling import stratified_client_ids
from src.data.loader import add_features, load_dataset
from src.experiments.artifacts import (
    atomic_write_json,
    file_sha256,
    fingerprint,
    git_revision,
)
from src.experiments.config_builder import load_yaml, write_runtime_config


def _distribution(transactions, selected: set[int]) -> dict[str, int]:
    clients = (
        transactions[transactions["customer_id"].isin(selected)]
        .groupby("customer_id")["label"]
        .first()
    )
    return {
        str(label): int(count)
        for label, count in clients.value_counts().sort_index().items()
    }


def configure(
    *,
    run_id: str,
    train_size: int,
    val_size: int,
    sampling_seed: int,
    execute: bool,
) -> dict[str, Any]:
    generated = Path("logs/runs") / run_id / "generated"
    selection_path = generated / "age_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    claimed_hash = selection.pop("selection_sha256", None)
    if (
        selection.get("status") != "completed"
        or selection.get("run_id") != run_id
        or selection.get("dataset") != "age"
        or not claimed_hash
        or fingerprint(selection) != claimed_hash
    ):
        raise RuntimeError(f"Invalid Age selection artifact: {selection_path}")

    base = load_yaml("configs/age.yaml")
    train = add_features(load_dataset(base, "train"))
    validation = add_features(load_dataset(base, "val"))
    train_ids = stratified_client_ids(
        train, n_clients=train_size, seed=sampling_seed
    )
    val_ids = stratified_client_ids(
        validation, n_clients=val_size, seed=sampling_seed
    )
    train_path = generated / "age_full_train_client_ids.json"
    val_path = generated / "age_full_val_client_ids.json"
    counts = {"train": len(train_ids), "val": len(val_ids), "test": 3000}
    result = {
        "mode": "execute" if execute else "dry-run",
        "run_id": run_id,
        "sampling_seed": sampling_seed,
        "expected_counts": counts,
        "train_label_distribution": _distribution(train, set(train_ids)),
        "val_label_distribution": _distribution(validation, set(val_ids)),
        "id_files": {"train": str(train_path), "val": str(val_path)},
        "models": sorted(selection.get("selected_configs", {})),
        "test_is_full": True,
    }
    if not execute:
        return result

    atomic_write_json(train_path, train_ids)
    atomic_write_json(val_path, val_ids)
    for model_slug, entry in selection["selected_configs"].items():
        config_path = Path(entry["path"])
        config = load_yaml(config_path)
        if Path(config["output"]["base_dir"]).exists():
            raise RuntimeError(
                f"Refusing to change Age selection after outputs exist: "
                f"{config['output']['base_dir']}"
            )
        config["dataset"]["client_ids_by_split"] = {
            "train": str(train_path),
            "val": str(val_path),
        }
        config["dataset"]["expected_client_counts"] = counts
        config.setdefault("pipeline", {})[
            "prompt_context_population"
        ] = "full_train_split"
        write_runtime_config(config_path, config)
        entry["sha256"] = file_sha256(config_path)

    selection["full_expected_client_counts"] = counts
    selection["full_run_sampling"] = {
        "method": (
            "stratified by label, transaction-count quartile, and "
            "absolute-transaction-volume quartile"
        ),
        "seed": sampling_seed,
        "client_ids_by_split": {
            "train": str(train_path),
            "val": str(val_path),
        },
        "client_id_sha256": {
            "train": file_sha256(train_path),
            "val": file_sha256(val_path),
        },
        "test_policy": "full split",
        "prompt_context_policy": "complete 24000-client train split",
        "future_expansion": (
            "same output root; add missing IDs after an explicit manifest "
            "expansion while reusing compatible completed records"
        ),
    }
    selection["git_revision"] = git_revision(REPO_ROOT)
    selection["selection_sha256"] = fingerprint(selection)
    atomic_write_json(selection_path, selection)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--sampling-seed", type=int, default=137)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.train_size < 1 or args.val_size < 1:
        raise ValueError("Subset sizes must be positive")
    print(
        json.dumps(
            configure(
                run_id=args.run_id,
                train_size=args.train_size,
                val_size=args.val_size,
                sampling_seed=args.sampling_seed,
                execute=args.execute,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

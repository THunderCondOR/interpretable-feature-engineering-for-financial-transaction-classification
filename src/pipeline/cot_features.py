"""Persist a train-only semantic feature space and frozen val/test transforms."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.pipeline.semantic_features import (
    claim_occurrences,
    fit_semantic_space,
    selected_view,
    transform_semantic_space,
)

SPLITS = ("train", "val", "test")


def split_output_path(config: dict, key: str, split: str) -> Path:
    out_dir, base = Path(config["output"]["base_dir"]), Path(config["output"][key])
    return out_dir / f"{base.stem}_{split}{base.suffix}"


def load_claim_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def flatten_claim_records(records):
    """Backwards-compatible view used by legacy analysis scripts."""
    rows = claim_occurrences(records)
    return (
        [row["normalized_text"] for row in rows],
        [row["label"] for row in rows],
        [row["customer_id"] for row in rows],
    )


def fit_train_clusters(config, train_records):
    return fit_semantic_space(config, train_records)


def transform_split(config, records, model):
    return transform_semantic_space(config, records, model)


def _json_model(model):
    return {
        key: value for key, value in model.items()
        if key not in {"centroids", "embedding_transformer"}
    }


def build_cot_features(config: dict) -> None:
    """This is an execution stage; orchestration decides whether it may run."""
    out_dir = Path(config["output"]["base_dir"])
    train_records = load_claim_records(split_output_path(config, "claims", "train"))
    model = fit_semantic_space(config, train_records)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "cot_clusters.json"
    meta_path.write_text(json.dumps(_json_model(model), ensure_ascii=False, indent=2), encoding="utf-8")
    model_path = out_dir / "cot_cluster_model.npz"
    np.savez_compressed(model_path, centroids=model["centroids"])
    if model.get("embedding_transformer") is not None:
        joblib.dump(model["embedding_transformer"], out_dir / "cot_embedding_model.joblib")
    print(f"Saved frozen train semantic space -> {meta_path}, {model_path}")

    for split in SPLITS:
        records = load_claim_records(split_output_path(config, "claims", split))
        features = transform_semantic_space(config, records, model)
        path = out_dir / f"cot_features_{split}.parquet"
        features.to_parquet(path, index=False)
        selected_path = out_dir / f"cot_features_selected_{split}.parquet"
        selected_view(features, model).to_parquet(selected_path, index=False)
        print(f"Saved {split} frozen features: shape={features.shape} -> {path}")

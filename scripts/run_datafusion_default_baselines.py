#!/usr/bin/env python3
"""Reproduce the official DF2023 RNN on the fixed local split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.common import load_json
from src.benchmarks.datafusion_teacher import extract_official_artifacts, predict_official
from src.experiments.artifacts import atomic_write_json, file_sha256, fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-zip", type=Path, default=Path("data/raw/datafusion_default_2023/model.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/isolated/baselines/datafusion_default_2023"))
    parser.add_argument("--seeds", default="17,101,947")
    parser.add_argument("--device")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    manifest = load_json(args.manifest)
    plan = {
        "mode": "execute" if args.execute else "dry-run", "benchmark": manifest.get("dataset"),
        "protocol": manifest.get("protocol"), "seeds": seeds,
        "published_reference": {"official_full_labeled_auc": 0.6839316224, "konderlip_cv_auc": 0.6804},
        "comparison_note": "Published CV numbers are context only; fixed-split metrics are computed here.",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if manifest.get("dataset") != "datafusion_default_2023":
        raise ValueError("Wrong benchmark manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = extract_official_artifacts(args.model_zip, args.output_dir / "official_model")
    events = pd.read_parquet(manifest["events"])
    labels = pd.read_csv(manifest["labels"], dtype={"customer_id": str})
    all_predictions, metrics = [], []
    for seed in seeds:
        predicted = predict_official(
            events, bins_path=artifacts["bins"], weights_path=artifacts["weights"],
            seed=seed, device=args.device,
        ).merge(labels, on="customer_id", validate="one_to_one")
        for split in ("train", "val", "test"):
            identifiers = set(json.loads(Path(manifest["roles"][split]).read_text()))
            cell = predicted[predicted["customer_id"].isin(identifiers)].copy()
            if len(cell) != int(manifest["counts"][split]):
                raise RuntimeError(f"Teacher coverage mismatch for {split}")
            cell["split"] = split
            all_predictions.append(cell)
            metrics.append({
                "teacher": "official_datafusion2023_rnn", "seed": seed, "split": split,
                "n_clients": len(cell), "roc_auc": float(roc_auc_score(cell["label"], cell["teacher_prob_1"])),
            })
    prediction_path = args.output_dir / "official_rnn_predictions.parquet"
    pd.concat(all_predictions, ignore_index=True).to_parquet(prediction_path, index=False)
    payload = {
        **plan, "status": "completed", "manifest_signature": manifest["manifest_signature"],
        "model_zip_sha256": file_sha256(args.model_zip), "prediction_sha256": file_sha256(prediction_path),
        "metrics": metrics,
        "stochasticity_note": "Final functional dropout remains active in the published model; seed is an experimental axis.",
    }
    payload["result_signature"] = fingerprint(payload)
    atomic_write_json(args.output_dir / "official_rnn_metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

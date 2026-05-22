"""
run_pipeline.py

Single entry point for the full pipeline.

Usage:
    python run_pipeline.py --config configs/gender.yaml --steps all
    python run_pipeline.py --config configs/age.yaml    --steps lora
    python run_pipeline.py --config configs/gender.yaml --steps stats,prompts,cot,claims,cluster,ml

Steps and their file dependencies:
    stats   → summary_stats.txt, clients_stats.jsonl
    prompts → reads summary_stats.txt → writes prompts.jsonl
    cot     → reads prompts.jsonl     → writes explanations.jsonl
    claims  → reads explanations.jsonl → writes claims.jsonl
    cluster → reads claims.jsonl      → writes cot_features.parquet
    lora    → trains LoRA model
    ml      → reads cot_features.parquet, builds handcrafted → writes ml_metrics.json
"""

import argparse
import json
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_dataset, add_features
from src.data.aggregator import build_dataset_summary_str, build_all_client_stats
from src.pipeline.prompt_builder import build_prompts, build_few_shot_str
from src.pipeline.explanation_gen import run_explanation_generation
from src.pipeline.claims_extractor import run_claims_extraction
from src.utils.cluster import embed_texts, fine_cluster_agglomerative
from src.utils.filtration import (
    filter_russian,
    clean_text,
    label_based_outlier_detection,
    filter_cluster_ids_by_class_diff,
)
from src.models.lora_trainer import train as lora_train
from src.models.ml_baseline import run_ml_baseline


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    # Normalize label_names keys to str (YAML may load bare integers as int keys)
    if "label_names" in config.get("dataset", {}):
        config["dataset"]["label_names"] = {
            str(k): v for k, v in config["dataset"]["label_names"].items()
        }
    return config


# ---------------------------------------------------------------------------
# Step: stats
# Builds dataset-level summary and per-client stats (for inspection).
# ---------------------------------------------------------------------------

def run_stats(config: dict) -> None:
    out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_df = add_features(load_dataset(config, "train"))
    val_df   = add_features(load_dataset(config, "val"))
    test_df  = add_features(load_dataset(config, "test"))

    print("Building dataset summary stats...")
    summary_str  = build_dataset_summary_str(train_df, config)
    summary_path = out_dir / config["output"]["summary_stats"]
    summary_path.write_text(summary_str, encoding="utf-8")
    print(f"  → {summary_path}")

    print("Building per-client stats...")
    eval_df        = pd.concat([val_df, test_df], ignore_index=True)
    client_records = build_all_client_stats(eval_df, config)

    clients_path = out_dir / config["output"]["clients_stats"]
    with open(clients_path, "w", encoding="utf-8") as f:
        for rec in client_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  → {clients_path}  ({len(client_records)} clients)")


# ---------------------------------------------------------------------------
# Step: prompts
# Builds full LLM prompts (system + user) and writes to prompts.jsonl.
# Must run after stats (reads summary_stats.txt).
# ---------------------------------------------------------------------------

def run_prompts(config: dict) -> None:
    out_dir = Path(config["output"]["base_dir"])

    train_df = add_features(load_dataset(config, "train"))
    val_df   = add_features(load_dataset(config, "val"))
    test_df  = add_features(load_dataset(config, "test"))

    summary_path = out_dir / config["output"]["summary_stats"]
    if not summary_path.exists():
        raise FileNotFoundError(
            f"summary_stats.txt not found at {summary_path}. "
            "Run --steps stats first."
        )
    summary_str  = summary_path.read_text(encoding="utf-8")
    few_shot_str = build_few_shot_str(train_df, config, n_per_class=1)
    eval_df      = pd.concat([val_df, test_df], ignore_index=True)
    records      = build_prompts(eval_df, config, summary_str, few_shot_str)

    # Write to prompts.jsonl (separate from clients_stats.jsonl)
    prompts_path = out_dir / config["output"]["prompts"]
    with open(prompts_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} prompts → {prompts_path}")


# ---------------------------------------------------------------------------
# Step: cot
# ---------------------------------------------------------------------------

def run_cot(config: dict) -> None:
    run_explanation_generation(config)


# ---------------------------------------------------------------------------
# Step: claims
# ---------------------------------------------------------------------------

def run_claims(config: dict) -> None:
    run_claims_extraction(config)


# ---------------------------------------------------------------------------
# Step: cluster
# Embed → remove outliers → cluster → filter by class separability → features
# ---------------------------------------------------------------------------

def run_cluster(config: dict) -> None:
    out_dir     = Path(config["output"]["base_dir"])
    claims_path = out_dir / config["output"]["claims"]

    records = []
    with open(claims_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    flat_claims = []
    flat_labels = []
    flat_cids   = []
    for rec in records:
        for claim in rec["claims"]:
            flat_claims.append(claim)
            flat_labels.append(rec["label"])
            flat_cids.append(rec["customer_id"])

    print(f"Loaded {len(flat_claims)} claims from {len(records)} clients")

    cleaned = [clean_text(c) for c in flat_claims]
    keep    = [bool(c) and bool(filter_russian([c])) for c in cleaned]
    flat_claims = [c for c, k in zip(cleaned,     keep) if k]
    flat_labels = [l for l, k in zip(flat_labels, keep) if k]
    flat_cids   = [c for c, k in zip(flat_cids,   keep) if k]
    print(f"After Russian filter: {len(flat_claims)} claims")

    embedding_model = config["pipeline"].get(
        "embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    print(f"Embedding claims with model: {embedding_model}")
    embeddings = embed_texts(flat_claims, model_name=embedding_model)

    outlier_flags = label_based_outlier_detection(embeddings, flat_labels, top_k=5)
    keep = [not flag for flag in outlier_flags]
    embeddings  = embeddings[np.array(keep)]
    flat_claims = [c for c, k in zip(flat_claims, keep) if k]
    flat_labels = [l for l, k in zip(flat_labels, keep) if k]
    flat_cids   = [c for c, k in zip(flat_cids,   keep) if k]
    print(f"After outlier removal: {len(flat_claims)} claims")

    dist_threshold = config["pipeline"].get("distance_threshold", 0.01)
    print(f"Clustering (distance_threshold={dist_threshold})...")
    cluster_ids = fine_cluster_agglomerative(embeddings, distance_threshold=dist_threshold)

    diff_threshold = config["pipeline"].get("class_diff_threshold", 0.02)
    num_labels     = config["dataset"]["num_labels"]
    cluster_ids = filter_cluster_ids_by_class_diff(
        cluster_ids, flat_labels, num_labels, min_diff=diff_threshold
    )
    n_kept = len(set(c for c in cluster_ids if c >= 0))
    print(f"After class-diff filter (>={diff_threshold}): {n_kept} clusters retained")

    unique_clusters = sorted(set(c for c in cluster_ids if c >= 0))
    cluster_to_idx  = {c: i for i, c in enumerate(unique_clusters)}
    n_features      = len(unique_clusters)

    cid_to_vec: dict = {rec["customer_id"]: np.zeros(n_features, dtype=np.float32)
                        for rec in records}
    for cid, clu in zip(flat_cids, cluster_ids):
        if clu >= 0 and clu in cluster_to_idx:
            cid_to_vec[cid][cluster_to_idx[clu]] += 1

    feat_records = [
        {
            "customer_id": rec["customer_id"],
            "label":       rec["label"],
            "features":    cid_to_vec[rec["customer_id"]].tolist(),
        }
        for rec in records
    ]

    feat_df   = pd.DataFrame(feat_records)
    feat_path = out_dir / config["output"].get("features", "cot_features.parquet")
    feat_df.to_parquet(str(feat_path), index=False)
    print(f"Saved {len(feat_df)} vectors ({n_features} dims) → {feat_path}")


# ---------------------------------------------------------------------------
# Step: lora
# ---------------------------------------------------------------------------

def run_lora(config: dict) -> None:
    lora_train(config)


# ---------------------------------------------------------------------------
# Step: ml
# ---------------------------------------------------------------------------

def run_ml(config: dict) -> None:
    run_ml_baseline(config)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

STEPS    = ["stats", "prompts", "cot", "claims", "cluster", "lora", "ml"]
STEP_FNS = {s: globals()[f"run_{s}"] for s in STEPS}


def main():
    parser = argparse.ArgumentParser(description="Interpretable Feature Engineering Pipeline")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    parser.add_argument(
        "--steps", default="all",
        help=f"Comma-separated steps or 'all'. Available: {', '.join(STEPS)}",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    steps  = STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]

    unknown = [s for s in steps if s not in STEP_FNS]
    if unknown:
        print(f"Unknown steps: {unknown}. Available: {list(STEP_FNS.keys())}", file=sys.stderr)
        sys.exit(1)

    for step in steps:
        print(f"\n{'='*55}\n  Step: {step}\n{'='*55}")
        STEP_FNS[step](config)

    print("\nDone.")


if __name__ == "__main__":
    main()
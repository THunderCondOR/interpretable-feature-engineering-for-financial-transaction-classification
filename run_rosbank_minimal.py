"""
run_rosbank_minimal.py

Fast Rosbank churn workflow for the experiments needed right now:
1) dataset-level and client-level aggregations,
2) majority baseline,
3) direct LLM baseline through OpenRouter/OpenAI-compatible API,
4) atomic-claim extraction,
5) train-only CoT feature construction.

Typical commands:
    python prepare_rosbank.py
    PYTHONPATH=. python run_rosbank_minimal.py --config configs/rosbank.yaml --steps stats,majority
    export OPENROUTER_API_KEY=...
    PYTHONPATH=. python run_rosbank_minimal.py --config configs/rosbank.yaml --steps prompts,llm,llm_eval --splits val --max-clients-per-split 200
    PYTHONPATH=. python run_rosbank_minimal.py --config configs/rosbank.yaml --steps claims --splits val

For CoT features you need claims for train and the target split:
    PYTHONPATH=. python run_rosbank_minimal.py --config configs/rosbank.yaml --steps prompts,llm,claims --splits train,val --max-clients-per-split 500
    PYTHONPATH=. python run_rosbank_minimal.py --config configs/rosbank.yaml --steps cot_features --splits train,val

LoRA remains available through:
    PYTHONPATH=. python run_lora.py --config configs/rosbank.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data.loader import load_dataset, add_features
from src.data.aggregator import build_dataset_summary_str, build_all_client_stats
from src.pipeline.prompt_builder import build_few_shot_str, build_prompts
from src.pipeline.explanation_gen import run_explanation_generation
from src.pipeline.claims_extractor import run_claims_extraction
from src.utils.filtration import clean_text, filter_cluster_ids_by_class_diff
from src.utils.cluster import fine_cluster_agglomerative


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if "label_names" in config.get("dataset", {}):
        config["dataset"]["label_names"] = {str(k): v for k, v in config["dataset"]["label_names"].items()}
    return config


def out_dir(config: dict) -> Path:
    path = Path(config["output"]["base_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_path(config: dict, key: str, split: str) -> Path:
    base = Path(config["output"][key])
    suffix = base.suffix or ".jsonl"
    return out_dir(config) / f"{base.stem}_{split}{suffix}"


def load_split(config: dict, split: str) -> pd.DataFrame:
    return add_features(load_dataset(config, split))


def maybe_sample_clients(df: pd.DataFrame, max_clients: int | None, seed: int = 42) -> pd.DataFrame:
    if not max_clients or df["customer_id"].nunique() <= max_clients:
        return df
    client_labels = df.drop_duplicates("customer_id")[["customer_id", "label"]]
    rng = np.random.default_rng(seed)
    # Stratified sampling if labels are available, random otherwise.
    if (client_labels["label"] >= 0).all() and client_labels["label"].nunique() > 1:
        sampled_ids = []
        per_label = max(1, max_clients // client_labels["label"].nunique())
        for _, group in client_labels.groupby("label"):
            ids = group["customer_id"].to_numpy()
            take = min(len(ids), per_label)
            sampled_ids.extend(rng.choice(ids, size=take, replace=False).tolist())
        if len(sampled_ids) < max_clients:
            rest = client_labels[~client_labels["customer_id"].isin(sampled_ids)]["customer_id"].to_numpy()
            if len(rest):
                sampled_ids.extend(rng.choice(rest, size=min(len(rest), max_clients - len(sampled_ids)), replace=False).tolist())
    else:
        ids = client_labels["customer_id"].to_numpy()
        sampled_ids = rng.choice(ids, size=max_clients, replace=False).tolist()
    return df[df["customer_id"].isin(set(sampled_ids))].copy()


def has_labels(df: pd.DataFrame) -> bool:
    return "label" in df.columns and (df.drop_duplicates("customer_id")["label"] >= 0).all()


def compute_classification_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except Exception as exc:
            metrics["roc_auc_error"] = str(exc)
    return metrics


def write_json(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def step_stats(config: dict, splits: Iterable[str]) -> None:
    od = out_dir(config)
    train_df = load_split(config, "train")
    summary = build_dataset_summary_str(train_df, config)
    summary_path = od / config["output"]["summary_stats"]
    summary_path.write_text(summary, encoding="utf-8")
    print(f"summary -> {summary_path}")

    report = {}
    for split in splits:
        df = load_split(config, split)
        records = build_all_client_stats(df, config)
        path = split_path(config, "clients_stats", split)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        labels = df.drop_duplicates("customer_id")["label"].value_counts().sort_index().to_dict()
        report[split] = {
            "rows": int(len(df)),
            "clients": int(df["customer_id"].nunique()),
            "labels_by_client": {str(k): int(v) for k, v in labels.items()},
        }
        print(f"client stats {split} -> {path} ({len(records)} clients)")
    write_json(od / "aggregation_report.json", report)


# ---------------------------------------------------------------------------
# majority baseline
# ---------------------------------------------------------------------------

def step_majority(config: dict, splits: Iterable[str]) -> None:
    train_df = load_split(config, "train")
    train_clients = train_df.drop_duplicates("customer_id")[["customer_id", "label"]]
    majority_label = int(train_clients["label"].value_counts().idxmax())
    majority_share = float((train_clients["label"] == majority_label).mean())

    results = {
        "majority_label": majority_label,
        "majority_label_name": config["dataset"]["label_names"].get(str(majority_label), str(majority_label)),
        "majority_share_train": round(majority_share, 4),
        "splits": {},
    }
    for split in splits:
        df = load_split(config, split)
        clients = df.drop_duplicates("customer_id")[["customer_id", "label"]]
        preds = np.full(len(clients), majority_label, dtype=int)
        pred_path = out_dir(config) / f"majority_predictions_{split}.csv"
        pd.DataFrame({"customer_id": clients.customer_id, "prediction": preds}).to_csv(pred_path, index=False)
        if has_labels(df):
            results["splits"][split] = compute_classification_metrics(clients["label"].values, preds)
        else:
            results["splits"][split] = {"n": int(len(clients)), "metrics_skipped": "labels are unavailable"}
        print(f"majority {split}: {results['splits'][split]}")
    write_json(out_dir(config) / "metrics_majority.json", results)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def step_prompts(config: dict, splits: Iterable[str], max_clients_per_split: int | None) -> None:
    od = out_dir(config)
    summary_path = od / config["output"]["summary_stats"]
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}; run stats first")
    summary = summary_path.read_text(encoding="utf-8")
    train_df = load_split(config, "train")
    few_shot = build_few_shot_str(train_df, config)
    (od / "few_shot_examples.txt").write_text(few_shot, encoding="utf-8")

    for split in splits:
        df = load_split(config, split)
        df = maybe_sample_clients(df, max_clients_per_split)
        records = build_prompts(df, config, summary, few_shot)
        path = split_path(config, "prompts", split)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"prompts {split} -> {path} ({len(records)} clients)")


# ---------------------------------------------------------------------------
# llm + llm eval
# ---------------------------------------------------------------------------

def step_llm(config: dict, splits: Iterable[str]) -> None:
    for split in splits:
        run_explanation_generation(config, split=split)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def step_llm_eval(config: dict, splits: Iterable[str]) -> None:
    results = {}
    for split in splits:
        path = split_path(config, "explanations", split)
        rows = _read_jsonl(path)
        by_client = defaultdict(list)
        labels = {}
        raw_answers = defaultdict(list)
        for r in rows:
            cid = int(r["customer_id"])
            by_client[cid].append(r.get("predicted"))
            raw_answers[cid].append(r.get("predicted_raw"))
            labels[cid] = int(r.get("label", -1))

        pred_records = []
        y_true, y_pred = [], []
        for cid, preds in by_client.items():
            valid = [p for p in preds if p is not None]
            pred = Counter(valid).most_common(1)[0][0] if valid else -1
            pred_records.append({
                "customer_id": cid,
                "label": labels[cid],
                "prediction": pred,
                "raw_answers": raw_answers[cid],
            })
            if labels[cid] >= 0 and pred >= 0:
                y_true.append(labels[cid])
                y_pred.append(pred)

        pred_path = out_dir(config) / f"llm_predictions_{split}.csv"
        pd.DataFrame(pred_records).to_csv(pred_path, index=False)
        if y_true:
            results[split] = compute_classification_metrics(y_true, y_pred)
        else:
            results[split] = {"n": len(pred_records), "metrics_skipped": "labels or valid predictions unavailable"}
        print(f"llm eval {split}: {results[split]}")
    write_json(out_dir(config) / "metrics_llm.json", results)


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------

def step_claims(config: dict, splits: Iterable[str]) -> None:
    for split in splits:
        run_claims_extraction(config, split=split)


# ---------------------------------------------------------------------------
# CoT features: fit clusters on train claims only, transform all requested splits.
# ---------------------------------------------------------------------------

def _load_claims(path: Path) -> list[dict]:
    return _read_jsonl(path)


def _flatten_claims(records: list[dict], require_labeled: bool) -> tuple[list[str], list[int], list[int]]:
    texts, labels, cids = [], [], []
    for rec in records:
        label = int(rec.get("label", -1))
        if require_labeled and label < 0:
            continue
        for claim in rec.get("claims", []):
            cleaned = clean_text(str(claim)).strip()
            if cleaned:
                texts.append(cleaned)
                labels.append(label)
                cids.append(int(rec["customer_id"]))
    return texts, labels, cids


def _embed(texts: list[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)


def _cluster_centroids(embeddings: np.ndarray, cluster_ids: np.ndarray) -> dict[int, np.ndarray]:
    centroids = {}
    for clu in sorted(set(int(c) for c in cluster_ids if c >= 0)):
        centroid = embeddings[cluster_ids == clu].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[clu] = centroid / norm if norm > 0 else centroid
    return centroids


def _make_feature_df(records: list[dict], claim_texts: list[str], claim_cids: list[int], claim_clusters: np.ndarray, clusters: list[int]) -> pd.DataFrame:
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    cid_to_vec = {int(r["customer_id"]): np.zeros(len(clusters), dtype=np.float32) for r in records}
    for cid, clu in zip(claim_cids, claim_clusters):
        if clu >= 0 and clu in cluster_to_idx and cid in cid_to_vec:
            cid_to_vec[int(cid)][cluster_to_idx[int(clu)]] += 1
    rows = []
    for rec in records:
        cid = int(rec["customer_id"])
        row = {"customer_id": cid, "label": int(rec.get("label", -1))}
        row.update({f"cot_cluster_{c}": float(cid_to_vec[cid][i]) for i, c in enumerate(clusters)})
        rows.append(row)
    return pd.DataFrame(rows)


def step_cot_features(config: dict, splits: Iterable[str]) -> None:
    od = out_dir(config)
    train_claims_path = split_path(config, "claims", "train")
    if not train_claims_path.exists():
        raise FileNotFoundError(f"Missing train claims: {train_claims_path}. Generate train claims first.")

    train_records = _load_claims(train_claims_path)
    train_texts, train_labels, train_cids = _flatten_claims(train_records, require_labeled=True)
    if not train_texts:
        raise ValueError("No train claims found")

    model_name = config.get("pipeline", {}).get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    print(f"Embedding {len(train_texts)} train claims with {model_name}")
    train_emb = _embed(train_texts, model_name)

    dist_threshold = config.get("pipeline", {}).get("distance_threshold", 0.01)
    print(f"Clustering train claims, distance_threshold={dist_threshold}")
    cluster_ids = fine_cluster_agglomerative(train_emb, distance_threshold=dist_threshold)

    min_cluster_size = config.get("pipeline", {}).get("min_cluster_size", 2)
    counts = Counter(cluster_ids.tolist())
    cluster_ids = np.array([c if counts[c] >= min_cluster_size else -1 for c in cluster_ids], dtype=np.int32)

    diff_threshold = config.get("pipeline", {}).get("class_diff_threshold", 0.02)
    cluster_ids = filter_cluster_ids_by_class_diff(cluster_ids, train_labels, config["dataset"]["num_labels"], min_diff=diff_threshold)
    clusters = sorted(set(int(c) for c in cluster_ids if c >= 0))
    if not clusters:
        raise ValueError("No clusters retained. Decrease class_diff_threshold or distance_threshold.")

    centroids = _cluster_centroids(train_emb, cluster_ids)
    train_feat_df = _make_feature_df(train_records, train_texts, train_cids, cluster_ids, clusters)
    train_feat_path = od / "cot_features_train.parquet"
    train_feat_df.to_parquet(train_feat_path, index=False)
    print(f"CoT features train -> {train_feat_path} shape={train_feat_df.shape}")

    centroid_matrix = np.vstack([centroids[c] for c in clusters])
    max_dist = config.get("pipeline", {}).get("max_assign_distance", 0.45)
    cluster_meta = []
    for c in clusters:
        examples = [t for t, clu in zip(train_texts, cluster_ids) if int(clu) == c][:10]
        label_counts = Counter([l for l, clu in zip(train_labels, cluster_ids) if int(clu) == c])
        cluster_meta.append({"cluster_id": c, "size": counts[c], "label_counts": dict(label_counts), "examples": examples})
    write_json(od / "cot_cluster_meta.json", cluster_meta)

    for split in splits:
        if split == "train":
            continue
        claims_path = split_path(config, "claims", split)
        if not claims_path.exists():
            print(f"skip {split}: missing {claims_path}")
            continue
        records = _load_claims(claims_path)
        texts, labels, cids = _flatten_claims(records, require_labeled=False)
        if texts:
            emb = _embed(texts, model_name)
            dists = cosine_distances(emb, centroid_matrix)
            nearest = dists.argmin(axis=1)
            nearest_dist = dists.min(axis=1)
            assigned = np.array([clusters[i] if d <= max_dist else -1 for i, d in zip(nearest, nearest_dist)], dtype=np.int32)
        else:
            assigned = np.array([], dtype=np.int32)
        feat_df = _make_feature_df(records, texts, cids, assigned, clusters)
        path = od / f"cot_features_{split}.parquet"
        feat_df.to_parquet(path, index=False)
        print(f"CoT features {split} -> {path} shape={feat_df.shape}")


STEPS = {
    "stats": step_stats,
    "majority": step_majority,
    "prompts": step_prompts,
    "llm": step_llm,
    "llm_eval": step_llm_eval,
    "claims": step_claims,
    "cot_features": step_cot_features,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rosbank.yaml")
    parser.add_argument("--steps", required=True, help="Comma-separated: stats,majority,prompts,llm,llm_eval,claims,cot_features")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split list")
    parser.add_argument("--max-clients-per-split", type=int, default=None, help="Quick debug/sample mode")
    args = parser.parse_args()

    config = load_config(args.config)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"Unknown steps: {unknown}. Available: {list(STEPS)}")

    for step in steps:
        print(f"\n{'='*60}\nStep: {step}\n{'='*60}")
        if step == "prompts":
            STEPS[step](config, splits, args.max_clients_per_split)
        else:
            STEPS[step](config, splits)


if __name__ == "__main__":
    main()

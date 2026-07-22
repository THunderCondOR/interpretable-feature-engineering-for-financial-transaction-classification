"""Run CoT claim-cluster stability experiments without new LLM calls.

The script reuses existing ``claims_{split}.jsonl`` files and reruns only the
claim clustering / feature construction stage under alternative clustering
settings. It is intended for reviewer-facing stability checks:

- different random subsamples of train claims (seed sensitivity);
- different agglomerative distance thresholds;
- optionally, fixed numbers of clusters;
- downstream performance of the resulting frozen train-derived representation.

Outputs are written under ``<output-base-dir>/cluster_stability`` and never
overwrite the main ``cot_features_*.parquet`` or ``ml_metrics.json`` files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from xgboost import XGBClassifier

from src.pipeline.cot_features import flatten_claim_records, load_claim_records
from src.utils.cluster import embed_texts


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Variant:
    name: str
    seed: int
    distance_threshold: float | None = None
    n_clusters: int | None = None


def split_output_path(config: dict[str, Any], key: str, split: str) -> Path:
    out_dir = Path(config["output"]["base_dir"])
    base = Path(config["output"][key])
    return out_dir / f"{base.stem}_{split}{base.suffix}"


def load_config(config_path: Path, output_base_dir: str | None) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if output_base_dir:
        config["output"]["base_dir"] = output_base_dir
    return config


def sample_indices(n: int, max_items: int | None, seed: int) -> np.ndarray:
    """Return seed-dependent fit indices.

    Agglomerative clustering has no random_state, but exact tie handling can be
    order-sensitive. We therefore use the seed both for optional subsampling and
    for fit-order shuffling. When max_items is 0/None, all claims are retained
    but their fit order is still seed-dependent.
    """
    rng = random.Random(seed)
    if max_items is None or max_items <= 0 or max_items >= n:
        idx = list(range(n))
    else:
        idx = rng.sample(range(n), max_items)
    rng.shuffle(idx)
    return np.array(idx, dtype=np.int64)


def fit_variant(
    embeddings: np.ndarray,
    labels: list[int],
    customer_ids: list[int],
    claims: list[str],
    config: dict[str, Any],
    variant: Variant,
    *,
    max_train_claims: int | None,
) -> dict[str, Any]:
    """Fit one clustering variant on train claims only."""
    idx = sample_indices(len(claims), max_train_claims, variant.seed)
    fit_embeddings = embeddings[idx]
    fit_labels = [labels[i] for i in idx]
    fit_customer_ids = [customer_ids[i] for i in idx]
    fit_claims = [claims[i] for i in idx]

    if variant.n_clusters is not None:
        n_clusters = min(max(2, int(variant.n_clusters)), len(fit_claims))
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
    else:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(variant.distance_threshold),
            metric="cosine",
            linkage="average",
        )

    raw_cluster_ids = clusterer.fit_predict(fit_embeddings)
    filtered_cluster_ids = np.asarray(raw_cluster_ids, dtype=np.int32)

    min_cluster_size = int(config.get("clustering", {}).get(
        "min_client_coverage", config["pipeline"].get("min_cluster_size", 1)
    ))
    kept_old_ids = [
        cluster_id
        for cluster_id in sorted(set(int(c) for c in filtered_cluster_ids if c >= 0))
        if len({cid for cid, assigned in zip(fit_customer_ids, filtered_cluster_ids) if assigned == cluster_id}) >= min_cluster_size
    ]
    if not kept_old_ids:
        raise ValueError(f"No clusters survived filtering for variant={variant.name}")

    centroids = []
    feature_names = []
    cluster_meta = []
    for new_idx, old_id in enumerate(kept_old_ids):
        mask = filtered_cluster_ids == old_id
        centroid = fit_embeddings[mask].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
        feature = f"cot_cluster_{new_idx:04d}"
        feature_names.append(feature)
        cluster_claims = [claim for claim, keep in zip(fit_claims, mask) if keep]
        cluster_labels = [label for label, keep in zip(fit_labels, mask) if keep]
        cluster_meta.append(
            {
                "feature": feature,
                "old_cluster_id": int(old_id),
                "size": int(mask.sum()),
                "label_counts": {
                    str(k): int(v)
                    for k, v in pd.Series(cluster_labels).value_counts().sort_index().to_dict().items()
                },
                "examples": cluster_claims[:10],
            }
        )

    return {
        "variant": variant.__dict__,
        "fit_claim_indices": idx.tolist(),
        "fit_customer_ids": fit_customer_ids,
        "centroids": np.vstack(centroids),
        "feature_names": feature_names,
        "cluster_meta": cluster_meta,
        "n_fit_claims": int(len(fit_claims)),
        "cluster_formation": "label_agnostic",
        "n_clusters": int(len(feature_names)),
    }


def assign_claims(
    embeddings: np.ndarray,
    customer_ids: list[int],
    model: dict[str, Any],
    max_distance: float,
) -> np.ndarray:
    if len(customer_ids) == 0:
        return np.zeros(0, dtype=np.int32)
    distances = cosine_distances(embeddings, model["centroids"])
    nearest = distances.argmin(axis=1).astype(np.int32)
    nearest_distance = distances.min(axis=1)
    nearest[nearest_distance > max_distance] = -1
    return nearest


def records_to_feature_frame(
    records: list[dict[str, Any]],
    claim_customer_ids: list[int],
    assignments: np.ndarray,
    model: dict[str, Any],
) -> pd.DataFrame:
    feature_names = model["feature_names"]
    vectors = {
        int(record["customer_id"]): np.zeros(len(feature_names), dtype=np.float32)
        for record in records
    }
    for cid, cluster_idx in zip(claim_customer_ids, assignments):
        if int(cluster_idx) >= 0:
            vectors[int(cid)][int(cluster_idx)] = 1.0

    rows = []
    for record in records:
        cid = int(record["customer_id"])
        row = {"customer_id": cid, "label": int(record["label"])}
        row.update({name: float(value) for name, value in zip(feature_names, vectors[cid])})
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: dict[str, Any], seed: int) -> dict[str, Any]:
    feature_cols = [c for c in train_df.columns if c not in {"customer_id", "label"}]
    x_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].astype(int).to_numpy()
    x_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["label"].astype(int).to_numpy()
    x_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].astype(int).to_numpy()

    objective = "binary:logistic" if int(config["dataset"]["num_labels"]) == 2 else "multi:softprob"
    model = XGBClassifier(
        objective=objective,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=2,
        tree_method="hist",
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)

    def metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        pred = model.predict(x)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        }

    return {
        "val": metrics(x_val, y_val),
        "test": metrics(x_test, y_test),
    }


def centroid_stability(reference: dict[str, Any], model: dict[str, Any]) -> dict[str, float]:
    sim = cosine_similarity(reference["centroids"], model["centroids"])
    return {
        "mean_ref_to_variant_max_cosine": float(sim.max(axis=1).mean()),
        "mean_variant_to_ref_max_cosine": float(sim.max(axis=0).mean()),
    }


def parse_variants(args: argparse.Namespace) -> list[Variant]:
    variants: list[Variant] = []
    for seed in args.seeds:
        for threshold in args.distance_thresholds:
            variants.append(
                Variant(
                    name=f"threshold_{threshold:g}_seed_{seed}",
                    seed=seed,
                    distance_threshold=threshold,
                )
            )
        for k in args.n_clusters:
            variants.append(
                Variant(
                    name=f"k_{k}_seed_{seed}",
                    seed=seed,
                    n_clusters=k,
                )
            )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-base-dir", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 101, 947])
    parser.add_argument("--distance-thresholds", type=float, nargs="+", default=[0.008, 0.01, 0.012])
    parser.add_argument("--n-clusters", type=int, nargs="*", default=[100, 200, 400, 800])
    parser.add_argument("--max-train-claims", type=int, default=0, help="0 means use all train claims.")
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps({"mode": "dry-run", "seeds": args.seeds,
            "distance_thresholds": args.distance_thresholds,
            "fixed_cluster_counts": args.n_clusters,
            "max_train_claims": args.max_train_claims or "all"}, indent=2))
        return

    config = load_config(args.config, args.output_base_dir)
    out_dir = Path(config["output"]["base_dir"]) / "cluster_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    records_by_split = {
        split: load_claim_records(split_output_path(config, "claims", split))
        for split in SPLITS
    }
    flattened = {}
    all_texts = []
    for split, records in records_by_split.items():
        claims, labels, customer_ids = flatten_claim_records(records)
        flattened[split] = {
            "claims": claims,
            "labels": labels,
            "customer_ids": customer_ids,
        }
        all_texts.extend(claims)

    embedding_model = config["pipeline"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    print(f"Embedding {len(all_texts)} claims with {embedding_model}")
    all_embeddings = embed_texts(all_texts, model_name=embedding_model)
    offset = 0
    for split in SPLITS:
        n = len(flattened[split]["claims"])
        flattened[split]["embeddings"] = all_embeddings[offset : offset + n]
        offset += n

    max_train_claims = args.max_train_claims if args.max_train_claims > 0 else None
    variants = parse_variants(args)
    if not variants:
        raise ValueError("No variants requested.")

    reference_model: dict[str, Any] | None = None
    reference_assignments: np.ndarray | None = None
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as file:
            rows = json.load(file)
    else:
        rows = []
    completed = {row.get("variant") for row in rows}

    for i, variant in enumerate(variants):
        variant_dir = out_dir / variant.name
        if variant.name in completed and (variant_dir / "cluster_meta.json").exists():
            print(f"\n=== Variant {i + 1}/{len(variants)}: {variant.name} already complete; skipping ===")
            continue
        print(f"\n=== Variant {i + 1}/{len(variants)}: {variant.name} ===")
        model = fit_variant(
            flattened["train"]["embeddings"],
            flattened["train"]["labels"],
            flattened["train"]["customer_ids"],
            flattened["train"]["claims"],
            config,
            variant,
            max_train_claims=max_train_claims,
        )

        max_distance = float(config["pipeline"].get("max_assign_distance", 0.45))
        assignments = {
            split: assign_claims(
                flattened[split]["embeddings"],
                flattened[split]["customer_ids"],
                model,
                max_distance=max_distance,
            )
            for split in SPLITS
        }
        features = {
            split: records_to_feature_frame(
                records_by_split[split],
                flattened[split]["customer_ids"],
                assignments[split],
                model,
            )
            for split in SPLITS
        }

        if reference_model is None:
            reference_model = model
            reference_assignments = assignments["train"]

        assert reference_model is not None
        assert reference_assignments is not None
        ari = adjusted_rand_score(reference_assignments, assignments["train"])
        nmi = normalized_mutual_info_score(reference_assignments, assignments["train"])
        centroid = centroid_stability(reference_model, model)

        xgb_metrics = None
        if not args.skip_xgb:
            xgb_metrics = evaluate_xgb(features["train"], features["val"], features["test"], config, variant.seed)

        row = {
            "variant": variant.name,
            "seed": variant.seed,
            "distance_threshold": variant.distance_threshold,
            "n_clusters_requested": variant.n_clusters,
            "n_fit_claims": model["n_fit_claims"],
            "cluster_formation": model["cluster_formation"],
            "n_clusters_kept": model["n_clusters"],
            "train_assignment_ari_vs_reference": float(ari),
            "train_assignment_nmi_vs_reference": float(nmi),
            **centroid,
        }
        if xgb_metrics is not None:
            row.update(
                {
                    "val_accuracy": xgb_metrics["val"]["accuracy"],
                    "val_balanced_accuracy": xgb_metrics["val"]["balanced_accuracy"],
                    "test_accuracy": xgb_metrics["test"]["accuracy"],
                    "test_balanced_accuracy": xgb_metrics["test"]["balanced_accuracy"],
                    "test_f1_macro": xgb_metrics["test"]["f1_macro"],
                }
            )
        variant_dir.mkdir(parents=True, exist_ok=True)
        with open(variant_dir / "cluster_meta.json", "w", encoding="utf-8") as file:
            json.dump(model["cluster_meta"], file, indent=2, ensure_ascii=False)
        for split, frame in features.items():
            frame.to_parquet(variant_dir / f"cot_features_{split}.parquet", index=False)

        rows = [existing for existing in rows if existing.get("variant") != variant.name]
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2, ensure_ascii=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
    print(f"\nSaved stability summary -> {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

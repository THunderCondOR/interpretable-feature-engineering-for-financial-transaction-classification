"""Post-hoc COFINFAD fidelity using transaction-only generated claims.

This is deliberately an offline diagnostic.  The source explanations were
generated from the full allowed operational profile, so the result must not be
described as a transaction-only prompt experiment.  We instead retain only
high-precision transaction claims, recluster their cached embeddings using
train data, and evaluate frozen val/test transforms.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from interpretable_features.experiments.artifacts import atomic_write_json, file_sha256, fingerprint

SPLITS = ("train", "val", "test")
MODELS = ("qwen", "gpt_oss")

# A claim must mention an explicit field available in the transaction-only
# profile.  Broad words such as "activity" and "engagement" are intentionally
# insufficient because they may refer to application/product usage.
TRANSACTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\btransactions?\b",
        r"\btransfers?\b",
        r"\bpayments?\b",
        r"\bwithdrawals?\b",
        r"\bdeposits?\b",
        r"\bactive days?\b",
        r"\bcalendar span\b",
        r"\bobservation (?:window|period|span)\b",
    )
)

# These phrases identify non-transactional operational evidence or an inferred
# outcome.  International/failed transactions are excluded because COFINFAD
# supplies them as customer-level operational fields, not event aggregates.
OPERATIONAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bproducts?\b",
        r"\baccounts?\b",
        r"\bcredit card\b",
        r"\bpersonal loan\b",
        r"\binvestment\b",
        r"\binsurance\b",
        r"\bauto(?:matic)?[- ]?savings\b",
        r"\bbill[- ]?payments?\b",
        r"\bapp(?:lication)?\b",
        r"\blogins?\b",
        r"\bfeature(?:s| use| usage| diversity| requests?)?\b",
        r"\bcredit utilization\b",
        r"\binternational transactions?\b",
        r"\bfailed transactions?\b",
        r"\bsatisfaction\b",
        r"\bnps\b|\bnet promoter\b",
        r"\bsupport\b|\btickets?\b",
        r"\bcomplaints?\b|\bcomplaint topics?\b",
        r"\bsentiment\b|\bfeedback\b|\bratings?\b",
        r"\brequests?\b|\brequested\b|\brequesting\b|\bfunctionality\b|\bmobile\b",
        r"\bservice\b|\bprovider\b|\bbank\b",
        r"\bchurn\b|\brisk\b|\bretention\b",
        r"\bloyal(?:ty)?\b|\bstick(?:y|iness)\b",
        r"\bdissatisf(?:ied|action)\b|\bfriction\b|\bstruggle\b",
        r"\brelationship\b|\bengagement\b",
    )
)


def is_transaction_claim(text: str) -> bool:
    """Return a conservative lexical transaction-only verdict."""
    value = str(text or "").strip()
    # LLM outputs frequently contain non-breaking/en/em hyphens.  Normalize
    # them before applying exclusions such as ``bill-payment`` or
    # ``auto-savings`` so typography cannot bypass the evidence boundary.
    value = re.sub(r"[\u2010-\u2015\u2212_-]+", " ", value)
    return (
        bool(value)
        and any(p.search(value) for p in TRANSACTION_PATTERNS)
        and not any(p.search(value) for p in OPERATIONAL_PATTERNS)
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "spearman_r": float(spearmanr(y_true, y_pred).statistic),
        "within_0_05": float((np.abs(y_true - y_pred) <= 0.05).mean()),
        "within_0_10": float((np.abs(y_true - y_pred) <= 0.10).mean()),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != "cofinfad_operational_fidelity":
        raise ValueError(f"Wrong benchmark manifest: {path}")
    return payload


def split_ids(manifest: dict, split: str) -> list[str]:
    values = json.loads(Path(manifest["roles"][split]).read_text(encoding="utf-8"))
    values = [str(value) for value in values]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {split} IDs")
    return values


def load_source(cell: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    assignments_path = cell / f"claim_assignments_{split}.parquet"
    unique_path = cell / "embeddings" / f"unique_claims_{split}.parquet"
    embeddings_path = cell / "embeddings" / f"embeddings_{split}.npy"
    assignments = pd.read_parquet(assignments_path)
    unique = pd.read_parquet(unique_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if len(unique) != len(embeddings):
        raise ValueError(f"Embedding row mismatch for {cell}/{split}")
    if unique["normalized_text"].duplicated().any():
        raise ValueError(f"Duplicate normalized unique claims for {cell}/{split}")
    assignments["customer_id"] = assignments["customer_id"].astype(str)
    assignments["normalized_text"] = assignments["normalized_text"].astype(str)
    assignments = assignments[assignments["assigned"].astype(bool)].copy()
    unique["normalized_text"] = unique["normalized_text"].astype(str)
    return assignments, unique, embeddings


def filtered_unique(
    unique: pd.DataFrame, embeddings: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = unique["normalized_text"].map(is_transaction_claim).to_numpy(dtype=bool)
    return unique.loc[mask].reset_index(drop=True), np.asarray(embeddings[mask], dtype=np.float32)


def clustered_occurrences(
    assignments: pd.DataFrame,
    unique: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    mapping = dict(zip(unique["normalized_text"], labels, strict=True))
    retained = assignments[assignments["normalized_text"].map(is_transaction_claim)].copy()
    retained["transaction_cluster"] = retained["normalized_text"].map(mapping)
    missing = int(retained["transaction_cluster"].isna().sum())
    if missing:
        raise ValueError(f"{missing} retained occurrences lack a cached embedding")
    retained["transaction_cluster"] = retained["transaction_cluster"].astype(int)
    return retained


def eligible_clusters(train: pd.DataFrame, min_clients: int) -> list[int]:
    coverage = train.groupby("transaction_cluster")["customer_id"].nunique()
    return sorted(int(value) for value in coverage[coverage >= min_clients].index)


def feature_matrix(
    occurrences: pd.DataFrame,
    ids: list[str],
    clusters: list[int],
    encoding: str,
) -> pd.DataFrame:
    if clusters:
        selected = occurrences[occurrences["transaction_cluster"].isin(clusters)]
        counts = pd.crosstab(selected["customer_id"], selected["transaction_cluster"])
        counts = counts.reindex(index=ids, columns=clusters, fill_value=0).astype(np.float32)
    else:
        counts = pd.DataFrame(index=ids, columns=[], dtype=np.float32)
    if encoding == "binary":
        counts = (counts > 0).astype(np.float32)
    elif encoding == "normalized_count":
        denominator = counts.sum(axis=1).replace(0, 1)
        counts = counts.div(denominator, axis=0).astype(np.float32)
    elif encoding != "raw_count":
        raise ValueError(f"Unknown encoding: {encoding}")
    counts.columns = [f"txn_clu_{value:04d}" for value in clusters]
    counts.insert(0, "customer_id", counts.index.astype(str))
    return counts.reset_index(drop=True)


def rank_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_y = y - y.mean()
    centered_x = x - x.mean(axis=0)
    denominator = np.sqrt(np.square(centered_x).sum(axis=0) * np.square(centered_y).sum())
    numerator = np.abs(centered_x.T @ centered_y)
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.argsort(-scores, kind="stable")


def evaluate_model(
    model_name: str,
    cell: Path,
    manifest: dict,
    output: Path,
    cluster_sizes: list[int],
    min_clients: int,
    seed: int,
) -> dict:
    ids = {split: split_ids(manifest, split) for split in SPLITS}
    scores = pd.read_csv(manifest["teacher_scores"], dtype={"customer_id": str})
    score_by_id = scores.set_index("customer_id")
    targets = {
        split: {
            "score": score_by_id.loc[ids[split], "churn_probability"].to_numpy(dtype=float),
            "label": score_by_id.loc[ids[split], "label"].to_numpy(dtype=int),
        }
        for split in SPLITS
    }
    source = {split: load_source(cell, split) for split in SPLITS}
    filtered = {split: filtered_unique(source[split][1], source[split][2]) for split in SPLITS}
    if not len(filtered["train"][0]):
        raise ValueError(f"No transaction-only claims for {model_name}")

    source_counts = {
        split: {
            "all_occurrences": int(len(source[split][0])),
            "transaction_occurrences": int(
                source[split][0]["normalized_text"].map(is_transaction_claim).sum()
            ),
            "all_unique": int(len(source[split][1])),
            "transaction_unique": int(len(filtered[split][0])),
        }
        for split in SPLITS
    }
    candidates: list[dict] = []
    materialized: dict[tuple[int, str], dict[str, pd.DataFrame]] = {}

    for requested_k in cluster_sizes:
        k = min(int(requested_k), len(filtered["train"][0]))
        if k < 2:
            continue
        clusterer = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=2048,
            n_init=3,
            max_iter=100,
            reassignment_ratio=0.01,
        )
        train_labels = clusterer.fit_predict(filtered["train"][1])
        occurrence_cells = {
            "train": clustered_occurrences(source["train"][0], filtered["train"][0], train_labels)
        }
        for split in ("val", "test"):
            predicted = clusterer.predict(filtered[split][1])
            occurrence_cells[split] = clustered_occurrences(
                source[split][0], filtered[split][0], predicted
            )
        clusters = eligible_clusters(occurrence_cells["train"], min_clients)
        if not clusters:
            continue
        for encoding in ("binary", "raw_count", "normalized_count"):
            frames = {
                split: feature_matrix(occurrence_cells[split], ids[split], clusters, encoding)
                for split in SPLITS
            }
            materialized[(requested_k, encoding)] = frames
            columns = [column for column in frames["train"] if column != "customer_id"]
            x_train = frames["train"][columns].to_numpy(dtype=np.float32)
            x_val = frames["val"][columns].to_numpy(dtype=np.float32)
            ranking = rank_features(x_train, targets["train"]["score"])
            for requested_top_k in (25, 50, 100, 200, None):
                n_keep = (
                    len(columns) if requested_top_k is None else min(requested_top_k, len(columns))
                )
                if n_keep < 1:
                    continue
                chosen = ranking[:n_keep]
                regressor = ExtraTreesRegressor(
                    n_estimators=250,
                    min_samples_leaf=5,
                    max_features=0.7,
                    random_state=seed,
                    n_jobs=-1,
                )
                regressor.fit(x_train[:, chosen], targets["train"]["score"])
                val_prediction = regressor.predict(x_val[:, chosen])
                candidates.append(
                    {
                        "requested_clusters": int(requested_k),
                        "fitted_clusters": int(k),
                        "clusters_after_coverage": int(len(clusters)),
                        "encoding": encoding,
                        "requested_top_k": requested_top_k,
                        "n_features": int(n_keep),
                        "selected_feature_indices": [int(value) for value in chosen],
                        "selected_feature_names": [columns[int(value)] for value in chosen],
                        "validation": regression_metrics(targets["val"]["score"], val_prediction),
                    }
                )

    if not candidates:
        raise ValueError(f"No valid ablation candidates for {model_name}")
    selected = sorted(
        candidates,
        key=lambda row: (
            -row["validation"]["r2"],
            row["validation"]["mae"],
            row["n_features"],
            row["requested_clusters"],
            row["encoding"],
        ),
    )[0]
    frames = materialized[(selected["requested_clusters"], selected["encoding"])]
    columns = [column for column in frames["train"] if column != "customer_id"]
    chosen = np.asarray(selected["selected_feature_indices"], dtype=int)
    x_train_val = np.vstack(
        [
            frames["train"][columns].to_numpy(dtype=np.float32)[:, chosen],
            frames["val"][columns].to_numpy(dtype=np.float32)[:, chosen],
        ]
    )
    y_train_val_score = np.concatenate([targets["train"]["score"], targets["val"]["score"]])
    y_train_val_label = np.concatenate([targets["train"]["label"], targets["val"]["label"]])
    x_test = frames["test"][columns].to_numpy(dtype=np.float32)[:, chosen]

    regressor = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=seed,
        n_jobs=-1,
    ).fit(x_train_val, y_train_val_score)
    classifier = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        max_features=0.7,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(x_train_val, y_train_val_label)
    test_score_prediction = np.clip(regressor.predict(x_test), 0.0, 1.0)
    test_label_prediction = classifier.predict(x_test)

    selected_frames = {}
    for split, frame in frames.items():
        selected_frame = frame[["customer_id", *selected["selected_feature_names"]]].copy()
        selected_frame.insert(1, "label", targets[split]["label"])
        path = output / model_name / f"transaction_claim_features_{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        selected_frame.to_parquet(path, index=False)
        selected_frames[split] = path

    payload = {
        "status": "completed",
        "experiment": "posthoc_transaction_claim_reclustering",
        "model": model_name,
        "source_cell": str(cell),
        "warning": (
            "Source explanations saw the allowed operational profile. This is a post-hoc "
            "transaction-claim ablation, not a transaction-only prompt experiment."
        ),
        "filter": {
            "method": "high_precision_lexical_whitelist_with_operational_exclusions",
            "transaction_patterns": [value.pattern for value in TRANSACTION_PATTERNS],
            "operational_patterns": [value.pattern for value in OPERATIONAL_PATTERNS],
        },
        "source_counts": source_counts,
        "selection_split": "val",
        "report_split": "test",
        "selected": selected,
        "test_continuous_fidelity": regression_metrics(
            targets["test"]["score"], test_score_prediction
        ),
        "test_quartile_classification": classification_metrics(
            targets["test"]["label"], test_label_prediction
        ),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "selected_feature_files": {
            split: {"path": str(path), "sha256": file_sha256(path)}
            for split, path in selected_frames.items()
        },
    }
    payload["result_signature"] = fingerprint(payload)
    atomic_write_json(output / model_name / "metrics.json", payload)
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--derived-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--cluster-sizes", default="100,200,400")
    parser.add_argument("--min-clients", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    cluster_sizes = [int(value) for value in args.cluster_sizes.split(",") if value.strip()]
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "experiment": "posthoc_transaction_claim_reclustering",
        "models": models,
        "cluster_sizes": cluster_sizes,
        "min_clients": args.min_clients,
        "source": str(args.derived_root),
        "output": str(args.output),
        "api_calls": 0,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    manifest = load_manifest(args.manifest)
    results = {}
    for model in models:
        cell = args.derived_root / model / "seed_17"
        results[model] = evaluate_model(
            model, cell, manifest, args.output, cluster_sizes, args.min_clients, args.seed
        )
        metric = results[model]["test_continuous_fidelity"]
        print(f"[{model}] test R2={metric['r2']:.4f} MAE={metric['mae']:.4f}")
    summary = {
        **plan,
        "mode": "completed",
        "results": {
            model: {
                "source_counts": payload["source_counts"],
                "selected": payload["selected"],
                "test_continuous_fidelity": payload["test_continuous_fidelity"],
                "test_quartile_classification": payload["test_quartile_classification"],
                "result_signature": payload["result_signature"],
            }
            for model, payload in results.items()
        },
    }
    summary["result_signature"] = fingerprint(summary)
    atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

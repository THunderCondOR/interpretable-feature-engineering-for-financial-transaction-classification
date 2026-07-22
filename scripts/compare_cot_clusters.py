"""Compare CoT claim clusters produced by two LLM runs.

The script uses existing ``cot_clusters.json`` files and embeds cluster example
claims to estimate cross-run semantic overlap. It does not call any LLM API and
does not modify experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.cluster import embed_texts


def load_clusters(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        payload = payload.get("cluster_meta", [])
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"Unsupported cluster artifact schema: {path}")
    return payload


def cluster_text(cluster: dict[str, Any], max_examples: int) -> str:
    examples = cluster.get("examples") or []
    examples = [str(x).strip() for x in examples[:max_examples] if str(x).strip()]
    if not examples:
        return str(cluster.get("feature", ""))
    return " ; ".join(examples)


def summarize_matches(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    sim: np.ndarray,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not left or not right:
        return {
            "n_left_clusters": len(left),
            "n_right_clusters": len(right),
            "mean_left_to_right_best_cosine": None,
            "mean_right_to_left_best_cosine": None,
            "one_to_one_mean_cosine": None,
            "mutual_nearest_pairs": 0,
            "mutual_nearest_share_left": 0.0,
            "unmatched_left_mass": 1.0 if left else 0.0,
            "unmatched_right_mass": 1.0 if right else 0.0,
        }, []
    left_best = sim.max(axis=1)
    right_best = sim.max(axis=0) if len(left) and len(right) else np.array([])
    weights_left = [cluster.get("occurrences", cluster.get("size", 1)) for cluster in left]
    weights_right = [cluster.get("occurrences", cluster.get("size", 1)) for cluster in right]
    # Reuse the already-computed cosine matrix for deterministic Hungarian and
    # mutual-nearest matching without re-embedding.
    from scipy.optimize import linear_sum_assignment
    row_ids, col_ids = linear_sum_assignment(-sim)
    left_nn, right_nn = sim.argmax(1), sim.argmax(0)
    mutual = {(i, int(left_nn[i])) for i in range(len(left)) if right_nn[left_nn[i]] == i}
    lw, rw = np.asarray(weights_left, dtype=float), np.asarray(weights_right, dtype=float)
    lw, rw = lw / max(lw.sum(), 1), rw / max(rw.sum(), 1)
    summary = {
        "n_left_clusters": int(len(left)),
        "n_right_clusters": int(len(right)),
        "mean_left_to_right_best_cosine": float(left_best.mean()) if len(left_best) else None,
        "median_left_to_right_best_cosine": float(np.median(left_best)) if len(left_best) else None,
        "share_left_best_ge_0_80": float((left_best >= 0.80).mean()) if len(left_best) else None,
        "share_left_best_ge_0_70": float((left_best >= 0.70).mean()) if len(left_best) else None,
        "mean_right_to_left_best_cosine": float(right_best.mean()) if len(right_best) else None,
        "median_right_to_left_best_cosine": float(np.median(right_best)) if len(right_best) else None,
        "share_right_best_ge_0_80": float((right_best >= 0.80).mean()) if len(right_best) else None,
        "share_right_best_ge_0_70": float((right_best >= 0.70).mean()) if len(right_best) else None,
        "one_to_one_mean_cosine": float(sim[row_ids, col_ids].mean()) if len(row_ids) else None,
        "mutual_nearest_pairs": int(len(mutual)),
        "mutual_nearest_share_left": float(len(mutual) / max(len(left), 1)),
        "size_weighted_left_best_cosine": float(np.average(left_best, weights=lw)) if len(left_best) else None,
        "size_weighted_right_best_cosine": float(np.average(right_best, weights=rw)) if len(right_best) else None,
        "unmatched_left_mass": float(1 - lw[row_ids].sum()) if len(row_ids) else 1.0,
        "unmatched_right_mass": float(1 - rw[col_ids].sum()) if len(col_ids) else 1.0,
    }

    pairs = []
    if len(left) and len(right):
        ranked_pairs = sorted(zip(row_ids, col_ids), key=lambda pair: sim[pair[0], pair[1]], reverse=True)[:top_k]
        for i, j in ranked_pairs:
            pairs.append(
                {
                    "left_feature": left[i].get("feature"),
                    "right_feature": right[j].get("feature"),
                    "cosine": float(sim[i, j]),
                    "mutual_nearest": (int(i), int(j)) in mutual,
                    "left_size": left[i].get("size"),
                    "right_size": right[j].get("size"),
                    "left_examples": (left[i].get("examples") or [])[:3],
                    "right_examples": (right[j].get("examples") or [])[:3],
                }
            )
    return summary, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-root", required=True, type=Path)
    parser.add_argument("--right-root", required=True, type=Path)
    parser.add_argument("--left-name", default="qwen")
    parser.add_argument("--right-name", default="gpt_oss")
    parser.add_argument("--datasets", nargs="+", default=["gender", "age", "rosbank"])
    parser.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "datasets": args.datasets,
        "left_root": str(args.left_root),
        "right_root": str(args.right_root),
        "output": str(args.output),
    }, indent=2))
    if not args.execute:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    details: dict[str, Any] = {}

    for dataset in args.datasets:
        left_path = args.left_root / dataset / "cot_clusters.json"
        right_path = args.right_root / dataset / "cot_clusters.json"
        left = load_clusters(left_path)
        right = load_clusters(right_path)
        texts = [cluster_text(c, args.max_examples) for c in left] + [
            cluster_text(c, args.max_examples) for c in right
        ]
        print(f"{dataset}: embedding {len(texts)} cluster texts")
        emb = embed_texts(texts, model_name=args.embedding_model)
        left_emb = emb[: len(left)]
        right_emb = emb[len(left) :]
        sim = cosine_similarity(left_emb, right_emb) if len(left) and len(right) else np.zeros((len(left), len(right)))
        summary, pairs = summarize_matches(left, right, sim, args.top_k)
        summary = {"dataset": dataset, **summary}
        rows.append(summary)
        details[dataset] = {
            "summary": summary,
            "top_pairs": pairs,
        }

    csv_path = args.output.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(
            {
                "left_name": args.left_name,
                "right_name": args.right_name,
                "left_root": str(args.left_root),
                "right_root": str(args.right_root),
                "details": details,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved cluster comparison -> {args.output}")
    print(f"Saved cluster comparison CSV -> {csv_path}")


if __name__ == "__main__":
    main()

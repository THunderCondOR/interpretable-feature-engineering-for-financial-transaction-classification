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
        return json.load(file)


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
    left_best = sim.max(axis=1) if len(left) and len(right) else np.array([])
    right_best = sim.max(axis=0) if len(left) and len(right) else np.array([])
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
    }

    pairs = []
    if len(left) and len(right):
        flat = np.argsort(sim.ravel())[::-1][:top_k]
        n_right = sim.shape[1]
        for flat_idx in flat:
            i, j = divmod(int(flat_idx), n_right)
            pairs.append(
                {
                    "left_feature": left[i].get("feature"),
                    "right_feature": right[j].get("feature"),
                    "cosine": float(sim[i, j]),
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
    args = parser.parse_args()

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

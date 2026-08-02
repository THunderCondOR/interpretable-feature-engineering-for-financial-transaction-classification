#!/usr/bin/env python3
"""Create namespaced metrics for two existing judges plus full GPT-5.5 judge."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_codex_grounding_full_judge import JUDGE_NAME, VERDICTS, read_jsonl
from scripts.run_grounding_judge import CLAIM_TYPES
from src.evaluation.reviewer_metrics import adjudicate_grounding
from src.experiments.artifacts import atomic_write_json


def load_and_validate(
    samples_path: Path,
    initial_paths: list[Path],
    codex_path: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    samples = read_jsonl(samples_path)
    sample_by_id = {str(row["sample_id"]): row for row in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("Duplicate grounding sample_id")

    initial = [row for path in initial_paths for row in read_jsonl(path)]
    codex = read_jsonl(codex_path)
    records = initial + codex
    if not records:
        raise ValueError("No grounding judgments supplied")
    frame = pd.DataFrame(records)
    required = {
        "sample_id", "judge_name", "evidence_hash", "verdict", "claim_type",
        "confidence", "evidence", "reason",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Grounding records missing columns: {sorted(missing)}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame.duplicated(["sample_id", "judge_name"], keep=False).any():
        raise ValueError("Duplicate judge verdict for a sample")
    if set(frame["sample_id"]) != set(sample_by_id):
        raise ValueError("Judge records do not exactly match grounding sample IDs")

    for sample_id, group in frame.groupby("sample_id", sort=False):
        names = group["judge_name"].astype(str).tolist()
        if len(names) != 3 or names.count(JUDGE_NAME) != 1:
            raise ValueError(
                f"Expected two initial judges and {JUDGE_NAME} for {sample_id}"
            )
        expected_hash = sample_by_id[sample_id].get("evidence_hash")
        hashes = group["evidence_hash"].tolist()
        if any(value != expected_hash for value in hashes):
            raise ValueError(f"Evidence hash mismatch for {sample_id}")
        if any(value not in VERDICTS for value in group["verdict"]):
            raise ValueError(f"Invalid verdict for {sample_id}")
        if any(value not in CLAIM_TYPES for value in group["claim_type"]):
            raise ValueError(f"Invalid claim_type for {sample_id}")
    return samples, frame


def categorical_majority(values: list[str]) -> tuple[str, bool]:
    counts = Counter(values).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "disagreement", False
    return counts[0][0], True


def fleiss_kappa(
    frame: pd.DataFrame,
    *,
    value_column: str,
    categories: list[str] | tuple[str, ...],
) -> float | None:
    """Fleiss' kappa for a complete fixed-size multi-judge rating matrix."""
    if frame.empty:
        return None
    table = pd.crosstab(frame["sample_id"], frame[value_column]).reindex(
        columns=list(categories), fill_value=0
    )
    ratings_per_item = table.sum(axis=1).to_numpy(dtype=float)
    if len(set(ratings_per_item.tolist())) != 1 or ratings_per_item[0] < 2:
        raise ValueError("Fleiss kappa requires equal judge coverage per item")
    n_ratings = float(ratings_per_item[0])
    counts = table.to_numpy(dtype=float)
    observed = ((counts * counts).sum(axis=1) - n_ratings) / (
        n_ratings * (n_ratings - 1.0)
    )
    category_prevalence = counts.sum(axis=0) / counts.sum()
    expected = float(np.square(category_prevalence).sum())
    if np.isclose(expected, 1.0):
        return None
    return float((observed.mean() - expected) / (1.0 - expected))


def cell_metrics(items: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (run_name, dataset), group in items.groupby(
        ["run_name", "dataset"], dropna=False, sort=True
    ):
        sample_ids = set(group["sample_id"].astype(str))
        judgments = frame[frame["sample_id"].astype(str).isin(sample_ids)]
        key = f"{run_name}/{dataset}"
        result[key] = {
            "run_name": run_name,
            "dataset": dataset,
            "n_items": int(len(group)),
            "n_clients": int(group["customer_id"].nunique(dropna=True)),
            "strict_consensus_share": float(group["strict_consensus"].mean()),
            "any_disagreement_share": float(group["has_disagreement"].mean()),
            "three_way_no_majority_share": float((~group["has_majority"]).mean()),
            "fleiss_kappa_verdict": fleiss_kappa(
                judgments, value_column="verdict", categories=VERDICTS
            ),
            "fleiss_kappa_claim_type": fleiss_kappa(
                judgments,
                value_column="claim_type",
                categories=tuple(sorted(CLAIM_TYPES)),
            ),
            "verdicts": {
                verdict: {
                    "count": int((group["final_verdict"] == verdict).sum()),
                    "share": float((group["final_verdict"] == verdict).mean()),
                }
                for verdict in (*VERDICTS, "disagreement")
            },
            "claim_types": {
                claim_type: {
                    "count": int((group["final_claim_type"] == claim_type).sum()),
                    "share": float((group["final_claim_type"] == claim_type).mean()),
                }
                for claim_type in (*sorted(CLAIM_TYPES), "disagreement")
            },
        }
    return result


def build_summary(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run_name, dataset), group in items.groupby(
        ["run_name", "dataset"], dropna=False, sort=True
    ):
        row: dict[str, Any] = {
            "run_name": run_name,
            "dataset": dataset,
            "n": int(len(group)),
            "strict_consensus_share": float(group["strict_consensus"].mean()),
            "any_disagreement_share": float(group["has_disagreement"].mean()),
            "no_majority_share": float((~group["has_majority"]).mean()),
        }
        for verdict in (*VERDICTS, "disagreement"):
            row[f"verdict_{verdict}_share"] = float(
                (group["final_verdict"] == verdict).mean()
            )
        for claim_type in (*sorted(CLAIM_TYPES), "disagreement"):
            row[f"claim_type_{claim_type}_share"] = float(
                (group["final_claim_type"] == claim_type).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_three_judges(
    samples: list[dict[str, Any]], frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        group = frame[frame["sample_id"] == sample_id]
        verdicts = group["verdict"].astype(str).tolist()
        final_verdict, has_majority = adjudicate_grounding(verdicts)
        claim_type, has_claim_type_majority = categorical_majority(
            group["claim_type"].astype(str).tolist()
        )
        rows.append({
            "sample_id": sample_id,
            "run_name": sample.get("run_name"),
            "dataset": sample.get("dataset"),
            "split": sample.get("split"),
            "customer_id": sample.get("customer_id"),
            "evidence_hash": sample.get("evidence_hash"),
            "final_verdict": final_verdict,
            "has_majority": has_majority,
            "strict_consensus": len(set(verdicts)) == 1,
            "has_disagreement": len(set(verdicts)) > 1,
            "final_claim_type": claim_type,
            "has_claim_type_majority": has_claim_type_majority,
            "judge_verdicts": json.dumps(dict(Counter(verdicts)), sort_keys=True),
            "judge_claim_types": json.dumps(
                dict(Counter(group["claim_type"].astype(str))), sort_keys=True
            ),
        })
    items = pd.DataFrame(rows)

    pairwise = []
    for left, right in combinations(sorted(frame["judge_name"].unique()), 2):
        pivot = frame[frame["judge_name"].isin([left, right])].pivot(
            index="sample_id", columns="judge_name", values="verdict"
        ).dropna()
        if pivot.empty:
            continue
        kappa = float(cohen_kappa_score(
            pivot[left], pivot[right], labels=list(VERDICTS)
        ))
        pairwise.append({
            "left": str(left),
            "right": str(right),
            "n": int(len(pivot)),
            "agreement": float((pivot[left] == pivot[right]).mean()),
            "cohen_kappa": kappa if np.isfinite(kappa) else None,
        })
    metrics: dict[str, Any] = {
        "protocol": "three_independent_judges",
        "codex_judge": JUDGE_NAME,
        "n_items": int(len(items)),
        "n_clients": int(items["customer_id"].nunique(dropna=True)),
        "strict_consensus_share": float(items["strict_consensus"].mean()),
        "any_disagreement_share": float(items["has_disagreement"].mean()),
        "three_way_no_majority_share": float((~items["has_majority"]).mean()),
        "fleiss_kappa_verdict": fleiss_kappa(
            frame, value_column="verdict", categories=VERDICTS
        ),
        "fleiss_kappa_claim_type": fleiss_kappa(
            frame,
            value_column="claim_type",
            categories=tuple(sorted(CLAIM_TYPES)),
        ),
        "pairwise_agreement": pairwise,
        "verdicts": {
            verdict: {
                "count": int((items["final_verdict"] == verdict).sum()),
                "share": float((items["final_verdict"] == verdict).mean()),
            }
            for verdict in (*VERDICTS, "disagreement")
        },
        "claim_types": {
            claim_type: {
                "count": int((items["final_claim_type"] == claim_type).sum()),
                "share": float((items["final_claim_type"] == claim_type).mean()),
            }
            for claim_type in (*sorted(CLAIM_TYPES), "disagreement")
        },
    }
    metrics["cells"] = cell_metrics(items, frame)
    return items, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--initial-inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--codex-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "samples": str(args.samples),
        "initial_inputs": [str(path) for path in args.initial_inputs],
        "codex_input": str(args.codex_input),
        "output_namespace": str(args.output_dir),
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return

    samples, frame = load_and_validate(
        args.samples, args.initial_inputs, args.codex_input
    )
    items, metrics = aggregate_three_judges(samples, frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_path = args.output_dir / "grounding_three_judge.items.csv"
    summary_path = args.output_dir / "grounding_three_judge.summary.csv"
    metrics_path = args.output_dir / "grounding_three_judge.metrics.json"
    items.to_csv(items_path, index=False)
    summary = build_summary(items)
    summary.to_csv(summary_path, index=False)
    atomic_write_json(metrics_path, metrics)
    print(f"Saved three-judge items -> {items_path}")
    print(f"Saved three-judge summary -> {summary_path}")
    print(f"Saved three-judge metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

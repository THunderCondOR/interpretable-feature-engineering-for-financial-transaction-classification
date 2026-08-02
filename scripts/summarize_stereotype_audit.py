#!/usr/bin/env python3
"""Validate and summarize two-judge stereotype-audit results."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_stereotype_audit_sample import read_jsonl
from scripts.run_stereotype_judge import VERDICTS


ORDERED_VERDICTS = [
    "no_sensitive_inference",
    "evidence_bounded_sensitive_inference",
    "weakly_grounded_sensitive_inference",
    "unsupported_stereotype",
    "not_assessable",
]
PROBLEMATIC = {
    "weakly_grounded_sensitive_inference", "unsupported_stereotype"
}


def finite_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2,
            default=lambda value: value.item()
            if isinstance(value, np.generic) else str(value),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(
                    value, ensure_ascii=False,
                    default=lambda item: item.item()
                    if isinstance(item, np.generic) else str(item),
                )
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })
    temporary.replace(path)


def validate_records(
    rows: list[dict[str, Any]], expected_judges: set[str]
) -> None:
    if not rows:
        raise ValueError("No stereotype judgments found")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["sample_id", "judge_name"]).any():
        raise ValueError("Duplicate sample/judge stereotype judgment")
    found = set(frame["judge_name"].astype(str))
    if found != expected_judges:
        raise ValueError(f"Judges mismatch: expected {expected_judges}, found {found}")
    counts = frame.groupby("sample_id")["judge_name"].nunique()
    if not (counts == len(expected_judges)).all():
        raise ValueError("Every sample must have exactly one result from each judge")
    if not set(frame["verdict"]).issubset(VERDICTS):
        raise ValueError("Invalid stereotype verdict in result records")
    for _, group in frame.groupby("sample_id"):
        for column in (
            "dataset", "customer_id", "run_name", "source_prompt_hash", "rationale"
        ):
            if group[column].astype(str).nunique() != 1:
                raise ValueError(f"Evidence mismatch between judges for {column}")


def rate_ci(values: list[bool], *, seed: int, samples: int = 3000) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0, "share": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    means = np.asarray([
        rng.choice(array, len(array), replace=True).mean() for _ in range(samples)
    ])
    return {
        "n": int(len(array)), "share": float(array.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def paired_delta(
    left: dict[str, bool], right: dict[str, bool], *, seed: int,
    samples: int = 3000,
) -> dict[str, Any]:
    clients = sorted(set(left) & set(right))
    if set(left) != set(right):
        raise ValueError("Source models do not use the same paired client IDs")
    differences = np.asarray([float(left[cid]) - float(right[cid]) for cid in clients])
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(differences, len(differences), replace=True).mean()
        for _ in range(samples)
    ])
    return {
        "n_paired_clients": len(clients),
        "delta_qwen_minus_gpt_oss": float(differences.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "qwen_problematic_gpt_clean": int(sum(left[cid] and not right[cid] for cid in clients)),
        "qwen_clean_gpt_problematic": int(sum(not left[cid] and right[cid] for cid in clients)),
    }


def build_summary(
    rows: list[dict[str, Any]], private_rows: list[dict[str, Any]], *, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    private = {str(row["sample_id"]): row for row in private_rows}
    enriched = []
    for row in rows:
        secret = private.get(str(row["sample_id"]))
        if secret is None:
            raise ValueError(f"Missing private metadata for {row['sample_id']}")
        enriched.append({**row, **{
            "true_label": secret["true_label"],
            "prediction_correct": secret["prediction_correct"],
            "is_problematic": row["verdict"] in PROBLEMATIC,
            "is_strict_unsupported": row["verdict"] == "unsupported_stereotype",
        }})
    frame = pd.DataFrame(enriched)
    judges = sorted(frame["judge_name"].unique())
    pivot = frame.pivot(index="sample_id", columns="judge_name", values="verdict")
    exact_agreement = float((pivot[judges[0]] == pivot[judges[1]]).mean())
    exact_kappa = finite_float(cohen_kappa_score(
        pivot[judges[0]], pivot[judges[1]], labels=ORDERED_VERDICTS
    ))
    ordinal_labels = ORDERED_VERDICTS[:-1]
    ordinal = pivot[
        pivot[judges[0]].isin(ordinal_labels) & pivot[judges[1]].isin(ordinal_labels)
    ]
    ordinal_kappa = finite_float(cohen_kappa_score(
        ordinal[judges[0]], ordinal[judges[1]], labels=ordinal_labels,
        weights="quadratic",
    )) if len(ordinal) else None
    binary_left = pivot[judges[0]].isin(PROBLEMATIC)
    binary_right = pivot[judges[1]].isin(PROBLEMATIC)
    binary_kappa = finite_float(cohen_kappa_score(binary_left, binary_right))

    cell_rows = []
    for keys, group in frame.groupby(["dataset", "run_name", "judge_name"], sort=True):
        verdict_counts = Counter(group["verdict"])
        flag_counts = Counter(flag for flags in group["flags"] for flag in flags)
        broad = rate_ci(group["is_problematic"].tolist(), seed=seed)
        strict = rate_ci(group["is_strict_unsupported"].tolist(), seed=seed + 1)
        cell_rows.append({
            "dataset": keys[0], "source_model": keys[1], "judge": keys[2],
            "n": len(group), "broad_problematic": broad,
            "strict_unsupported": strict,
            "verdict_counts": dict(verdict_counts), "flag_counts": dict(flag_counts),
        })

    item_rows = []
    priority_rows = []
    for sample_id, group in frame.groupby("sample_id", sort=False):
        verdicts = group["verdict"].tolist()
        first = group.iloc[0]
        exact = len(set(verdicts)) == 1
        any_problematic = any(value in PROBLEMATIC for value in verdicts)
        all_problematic = all(value in PROBLEMATIC for value in verdicts)
        item = {
            "sample_id": sample_id, "dataset": first["dataset"],
            "source_model": first["run_name"], "customer_id": first["customer_id"],
            "true_label": first["true_label"],
            "prediction_correct": bool(first["prediction_correct"]),
            "judge_verdicts": verdicts, "strict_consensus": exact,
            "has_disagreement": not exact,
            "any_judge_problematic": any_problematic,
            "both_judges_problematic": all_problematic,
        }
        item_rows.append(item)
        if any_problematic or not exact:
            priority_rows.append({
                **item,
                "rationale": first["rationale"],
                "judge_details": [{
                    "judge": row["judge_name"], "verdict": row["verdict"],
                    "flags": row["flags"], "quoted_span": row["quoted_span"],
                    "reason": row["reason"],
                } for _, row in group.iterrows()],
            })

    item_frame = pd.DataFrame(item_rows)
    consensus_cells = []
    for keys, group in item_frame.groupby(["dataset", "source_model"], sort=True):
        consensus_cells.append({
            "dataset": keys[0], "source_model": keys[1], "n": int(len(group)),
            "exact_agreement_share": float(group["strict_consensus"].mean()),
            "lower_both_judges_problematic": float(
                group["both_judges_problematic"].mean()
            ),
            "upper_any_judge_problematic": float(
                group["any_judge_problematic"].mean()
            ),
        })
    paired = []
    for dataset in sorted(frame["dataset"].unique()):
        for judge in judges:
            group = frame[(frame["dataset"] == dataset) & (frame["judge_name"] == judge)]
            models = set(group["run_name"])
            if models != {"qwen", "gpt_oss"}:
                raise ValueError(f"Expected paired qwen/gpt_oss for {dataset}/{judge}")
            values = {
                model: {
                    str(row["customer_id"]): bool(row["is_problematic"])
                    for _, row in group[group["run_name"] == model].iterrows()
                }
                for model in models
            }
            paired.append({
                "dataset": dataset, "judge": judge,
                **paired_delta(values["qwen"], values["gpt_oss"], seed=seed),
            })
    summary = {
        "n_rationales": int(frame["sample_id"].nunique()),
        "n_judgments": int(len(frame)),
        "n_unique_clients": int(frame[["dataset", "customer_id"]].drop_duplicates().shape[0]),
        "judges": judges,
        "agreement": {
            "exact_share": exact_agreement,
            "exact_cohen_kappa": exact_kappa,
            "ordinal_quadratic_kappa": ordinal_kappa,
            "binary_problematic_kappa": binary_kappa,
            "disagreement_share": float(1.0 - exact_agreement),
        },
        "problematic_bounds": {
            "lower_both_judges": float(item_frame["both_judges_problematic"].mean()),
            "upper_any_judge": float(item_frame["any_judge_problematic"].mean()),
        },
        "cells": cell_rows,
        "consensus_cells": consensus_cells,
        "paired_source_model_comparisons": paired,
        "priority_review_items": len(priority_rows),
        "note": "Two-judge disagreements are reported, never converted into a majority.",
    }
    return summary, item_rows, priority_rows


def report_markdown(summary: dict[str, Any]) -> str:
    agreement = summary["agreement"]
    lines = [
        "# Stereotype audit", "",
        f"Rationales: {summary['n_rationales']}; judgments: {summary['n_judgments']}; "
        f"paired clients: {summary['n_unique_clients']}.", "",
        "## Judge agreement", "",
        f"- Exact agreement: {agreement['exact_share']:.1%}",
        f"- Exact Cohen's kappa: {agreement['exact_cohen_kappa']:.3f}"
        if agreement["exact_cohen_kappa"] is not None else "- Exact Cohen's kappa: undefined",
        f"- Ordinal quadratic kappa: {agreement['ordinal_quadratic_kappa']:.3f}"
        if agreement["ordinal_quadratic_kappa"] is not None else "- Ordinal kappa: unavailable",
        f"- Binary problematic/non-problematic kappa: {agreement['binary_problematic_kappa']:.3f}"
        if agreement["binary_problematic_kappa"] is not None else "- Binary problematic/non-problematic kappa: undefined",
        "", "## Per-judge estimates", "",
        "| Dataset | Source | Judge | N | Broad problematic | 95% CI | Strict unsupported | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cells"]:
        broad, strict = row["broad_problematic"], row["strict_unsupported"]
        lines.append(
            f"| {row['dataset']} | {row['source_model']} | {row['judge']} | {row['n']} | "
            f"{broad['share']:.1%} | [{broad['ci_low']:.1%}, {broad['ci_high']:.1%}] | "
            f"{strict['share']:.1%} | [{strict['ci_low']:.1%}, {strict['ci_high']:.1%}] |"
        )
    lines.extend(["", "## Paired source-model differences", "",
        "Positive delta means a higher broad-problematic rate for Qwen.", "",
        "| Dataset | Judge | N | Qwen−GPT-OSS | 95% CI |",
        "|---|---|---:|---:|---:|",
    ])
    for row in summary["paired_source_model_comparisons"]:
        lines.append(
            f"| {row['dataset']} | {row['judge']} | {row['n_paired_clients']} | "
            f"{row['delta_qwen_minus_gpt_oss']:+.1%} | "
            f"[{row['ci_low']:+.1%}, {row['ci_high']:+.1%}] |"
        )
    lines.extend(["", summary["note"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-judges", nargs="+", required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "inputs": [str(path) for path in args.inputs],
        "expected_judges": args.expected_judges,
        "output_dir": str(args.output_dir),
    }, indent=2))
    if not args.execute:
        return
    rows = [row for path in args.inputs for row in read_jsonl(path)]
    validate_records(rows, set(args.expected_judges))
    private = json.loads(args.private_key.read_text(encoding="utf-8"))
    summary, items, priority = build_summary(rows, private, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "metrics.json", summary)
    write_csv(args.output_dir / "judgments.csv", rows)
    write_csv(args.output_dir / "items.csv", items)
    write_csv(args.output_dir / "priority_review.csv", priority)
    atomic_json(args.output_dir / "priority_review.json", priority)
    (args.output_dir / "report.md").write_text(
        report_markdown(summary), encoding="utf-8"
    )
    print(f"Saved stereotype audit report -> {args.output_dir}")


if __name__ == "__main__":
    main()

"""Summarize multi-judge grounding results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.reviewer_metrics import GROUNDING_VERDICTS, adjudicate_grounding, grounding_summary


VERDICTS = [*GROUNDING_VERDICTS, "disagreement"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def majority(verdicts: list[str]) -> tuple[str, bool]:
    return adjudicate_grounding(verdicts)


def validate_judge_records(df: pd.DataFrame, expected_judges: set[str]) -> None:
    required = {"sample_id", "judge_name", "evidence_hash", "verdict"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Grounding records missing columns: {sorted(missing_columns)}")
    if df.duplicated(["sample_id", "judge_name"], keep=False).any():
        raise ValueError("Duplicate judge verdicts for the same sample_id")
    for sample_id, group in df.groupby("sample_id", sort=False):
        actual = set(group["judge_name"].dropna().astype(str))
        if actual != expected_judges:
            raise ValueError(
                f"Incomplete judge set for sample_id={sample_id}: "
                f"expected={sorted(expected_judges)}, actual={sorted(actual)}"
            )
        hashes = set(group["evidence_hash"].dropna().astype(str))
        if len(hashes) != 1 or group["evidence_hash"].isna().any():
            raise ValueError(f"Evidence hash mismatch for sample_id={sample_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "inputs": [str(path) for path in args.inputs],
            "output_prefix": str(args.output_prefix),
        }, indent=2))
        return

    records = []
    expected_judges = set()
    for path in args.inputs:
        input_records = read_jsonl(path)
        judges = {str(row.get("judge_name")) for row in input_records if row.get("judge_name")}
        if len(judges) != 1:
            raise ValueError(f"Each judge input must contain exactly one judge: {path}")
        judge = next(iter(judges))
        if judge in expected_judges:
            raise ValueError(f"Duplicate judge input: {judge}")
        expected_judges.add(judge)
        records.extend(input_records)
    df = pd.DataFrame(records)
    validate_judge_records(df, expected_judges)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    grouped_rows = []
    disagreements = []
    for sample_id, group in df.groupby("sample_id"):
        verdicts = group["verdict"].fillna("missing").tolist()
        maj, has_majority = majority(verdicts)
        first = group.iloc[0].to_dict()
        row = {
            "sample_id": sample_id,
            "run_name": first.get("run_name"),
            "dataset": first.get("dataset"),
            "split": first.get("split"),
            "customer_id": first.get("customer_id"),
            "claim": first.get("claim"),
            "majority_verdict": maj,
            "has_majority": has_majority,
            "strict_consensus": len(set(verdicts)) == 1,
            "has_disagreement": len(set(verdicts)) > 1,
            "needs_adjudication": len(set(verdicts)) > 1 or maj == "unsupported",
            "n_judges": int(len(group)),
            "judge_verdicts": dict(Counter(verdicts)),
        }
        grouped_rows.append(row)
        if row["needs_adjudication"]:
            disagreements.append(
                {
                    **row,
                    "judge_details": group[
                        ["judge_name", "verdict", "confidence", "evidence", "reason"]
                    ].to_dict(orient="records"),
                }
            )

    item_df = pd.DataFrame(grouped_rows)
    summary_rows = []
    aggregate_metrics = {}
    for keys, group in item_df.groupby(["run_name", "dataset"], dropna=False):
        run_name, dataset = keys
        total = len(group)
        row = {"run_name": run_name, "dataset": dataset, "n": int(total)}
        for verdict in VERDICTS + ["missing"]:
            row[verdict] = int((group["majority_verdict"] == verdict).sum())
            row[f"{verdict}_share"] = float((group["majority_verdict"] == verdict).mean()) if total else 0.0
        row["strict_consensus_share"] = float(group["strict_consensus"].mean()) if total else 0.0
        row["disagreement_share"] = float(group["has_disagreement"].mean()) if total else 0.0
        summary_rows.append(row)
        source = df[(df["run_name"] == run_name) & (df["dataset"] == dataset)]
        _, metrics = grounding_summary(source)
        aggregate_metrics[f"{run_name}/{dataset}"] = metrics

    item_path = args.output_prefix.with_suffix(".items.csv")
    summary_path = args.output_prefix.with_suffix(".summary.csv")
    disagreement_path = args.output_prefix.with_suffix(".disagreements.json")
    metrics_path = args.output_prefix.with_suffix(".metrics.json")
    item_df.to_csv(item_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    with open(disagreement_path, "w", encoding="utf-8") as file:
        json.dump(disagreements, file, indent=2, ensure_ascii=False)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(aggregate_metrics, file, indent=2, ensure_ascii=False)
    print(f"Saved item-level grounding -> {item_path}")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved disagreements -> {disagreement_path}")
    print(f"Saved kappa/bootstrap metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

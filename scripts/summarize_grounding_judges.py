"""Summarize multi-judge grounding results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


VERDICTS = ["supported", "partially_supported", "unsupported", "not_applicable", "parse_error"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def majority(verdicts: list[str]) -> tuple[str, bool]:
    counts = Counter(verdicts)
    if not counts:
        return "missing", False
    top = counts.most_common()
    tied = len(top) > 1 and top[0][1] == top[1][1]
    return top[0][0], not tied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    records = []
    for path in args.inputs:
        records.extend(read_jsonl(path))
    df = pd.DataFrame(records)
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
            "n_judges": int(len(group)),
            "judge_verdicts": dict(Counter(verdicts)),
        }
        grouped_rows.append(row)
        if len(set(verdicts)) > 1:
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
    for keys, group in item_df.groupby(["run_name", "dataset"], dropna=False):
        run_name, dataset = keys
        total = len(group)
        row = {"run_name": run_name, "dataset": dataset, "n": int(total)}
        for verdict in VERDICTS + ["missing"]:
            row[verdict] = int((group["majority_verdict"] == verdict).sum())
            row[f"{verdict}_share"] = float((group["majority_verdict"] == verdict).mean()) if total else 0.0
        row["disagreement_share"] = float((~group["has_majority"]).mean()) if total else 0.0
        summary_rows.append(row)

    item_path = args.output_prefix.with_suffix(".items.csv")
    summary_path = args.output_prefix.with_suffix(".summary.csv")
    disagreement_path = args.output_prefix.with_suffix(".disagreements.json")
    item_df.to_csv(item_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    with open(disagreement_path, "w", encoding="utf-8") as file:
        json.dump(disagreements, file, indent=2, ensure_ascii=False)
    print(f"Saved item-level grounding -> {item_path}")
    print(f"Saved summary -> {summary_path}")
    print(f"Saved disagreements -> {disagreement_path}")


if __name__ == "__main__":
    main()

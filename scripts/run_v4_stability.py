#!/usr/bin/env python3
"""Measure v4 granularity stability from cached candidate artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.experiments.artifacts import atomic_write_json


def aligned_assignments(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    columns = ["claim_id", "cluster_index", "assigned"]
    merged = reference[columns].merge(
        candidate[columns],
        on="claim_id",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("Stability candidates use different claim anchor sets")
    return (
        merged["cluster_index_reference"].to_numpy(np.int32),
        merged["cluster_index_candidate"].to_numpy(np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-cell", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    selection_path = args.derived_cell / "cluster_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing cluster selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = [row["candidate"] for row in selection["candidates"]]
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "axis": "cluster_granularity",
        "derived_cell": str(args.derived_cell),
        "reference": selection["selected_candidate"],
        "candidates": candidates,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return

    reference_name = selection["selected_candidate"]
    reference = pd.read_parquet(
        args.derived_cell
        / "candidates"
        / reference_name
        / "claim_assignments_train.parquet"
    )
    rows = []
    for candidate in candidates:
        candidate_dir = args.derived_cell / "candidates" / candidate
        assignments = pd.read_parquet(
            candidate_dir / "claim_assignments_train.parquet"
        )
        left, right = aligned_assignments(reference, assignments)
        joint = (left >= 0) & (right >= 0)
        if int(joint.sum()) >= 2:
            ari = float(adjusted_rand_score(left[joint], right[joint]))
            nmi = float(normalized_mutual_info_score(left[joint], right[joint]))
        else:
            ari = nmi = float("nan")
        candidate_stage = json.loads(
            (
                args.derived_cell
                / "stages"
                / f"candidate_{candidate}.json"
            ).read_text(encoding="utf-8")
        )
        rows.append({
            "axis": "cluster_granularity",
            "candidate": candidate,
            "reference": reference_name,
            "n_clusters": candidate_stage["metrics"]["n_clusters"],
            "validation_balanced_accuracy": candidate_stage["metrics"][
                "validation_balanced_accuracy"
            ],
            "train_assignment_ari_vs_selected": ari,
            "train_assignment_nmi_vs_selected": nmi,
            "train_joint_assignment_coverage": float(joint.mean()),
            "train_candidate_assignment_coverage": float((right >= 0).mean()),
        })
    output = args.output or args.derived_cell / "stability" / "granularity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, {
        "protocol": plan,
        "rows": rows,
        "note": "Validation-only granularity sensitivity; test remains reserved for the selected main configuration.",
    })
    frame = pd.DataFrame(rows)
    csv_path = output.with_suffix(".csv")
    temporary = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(csv_path)
    print(f"Saved v4 granularity stability -> {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a paired legacy-versus-v4 stereotype-audit comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_stereotype_audit_sample import read_jsonl
from src.experiments.artifacts import atomic_write_json


PROBLEMATIC = {
    "weakly_grounded_sensitive_inference", "unsupported_stereotype"
}


def load_audit(root: Path, condition: str) -> list[dict[str, Any]]:
    private = {
        str(row["sample_id"]): row
        for row in json.loads(
            (root / "stereotype_samples.private.json").read_text(encoding="utf-8")
        )
    }
    paths = sorted(
        path for path in root.glob("judge_*.jsonl")
        if ".repair_attempts" not in path.name and ".unresolved" not in path.name
    )
    if len(paths) != 2:
        raise ValueError(f"Expected two completed judge files in {root}, found {paths}")
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            secret = private.get(str(row["sample_id"]))
            if secret is None:
                raise ValueError(f"Missing private key for {row['sample_id']}")
            rows.append({
                **row, "condition": condition,
                "true_label": secret["true_label"],
                "prediction_correct": secret["prediction_correct"],
            })
    return rows


def paired_binary_delta(
    before: dict[str, bool], after: dict[str, bool], *, seed: int,
    samples: int = 5000,
) -> dict[str, Any]:
    if set(before) != set(after):
        missing_before = set(after) - set(before)
        missing_after = set(before) - set(after)
        raise ValueError(
            f"Before/after client mismatch: before_missing={len(missing_before)}, "
            f"after_missing={len(missing_after)}"
        )
    clients = sorted(before)
    differences = np.asarray([
        float(before[cid]) - float(after[cid]) for cid in clients
    ])
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(differences, len(differences), replace=True).mean()
        for _ in range(samples)
    ])
    return {
        "n": len(clients),
        "legacy_share": float(np.mean([before[cid] for cid in clients])),
        "v4_share": float(np.mean([after[cid] for cid in clients])),
        "improvement_legacy_minus_v4": float(differences.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "improved_clients": int(sum(before[cid] and not after[cid] for cid in clients)),
        "regressed_clients": int(sum(not before[cid] and after[cid] for cid in clients)),
    }


def compare(before: list[dict], after: list[dict], *, seed: int) -> dict[str, Any]:
    judges_before = {str(row["judge_name"]) for row in before}
    judges_after = {str(row["judge_name"]) for row in after}
    if judges_before != judges_after or len(judges_before) != 2:
        raise ValueError("Before/after audits must use the same two judges")
    cells = []
    datasets = sorted({str(row["dataset"]) for row in before + after})
    models = sorted({str(row["run_name"]) for row in before + after})
    for dataset in datasets:
        for model in models:
            for judge in sorted(judges_before):
                def values(rows: list[dict], predicate) -> dict[str, bool]:
                    return {
                        str(row["customer_id"]): bool(predicate(row))
                        for row in rows
                        if row["dataset"] == dataset
                        and row["run_name"] == model
                        and row["judge_name"] == judge
                    }
                broad = paired_binary_delta(
                    values(before, lambda row: row["verdict"] in PROBLEMATIC),
                    values(after, lambda row: row["verdict"] in PROBLEMATIC),
                    seed=seed,
                )
                strict = paired_binary_delta(
                    values(before, lambda row: row["verdict"] == "unsupported_stereotype"),
                    values(after, lambda row: row["verdict"] == "unsupported_stereotype"),
                    seed=seed + 1,
                )
                flag_names = sorted({
                    flag for row in before + after
                    if row["dataset"] == dataset and row["run_name"] == model
                    and row["judge_name"] == judge
                    for flag in row.get("flags", []) if flag != "none"
                })
                flags = {
                    flag: paired_binary_delta(
                        values(before, lambda row, value=flag: value in row.get("flags", [])),
                        values(after, lambda row, value=flag: value in row.get("flags", [])),
                        seed=seed + index + 10,
                    )
                    for index, flag in enumerate(flag_names)
                }
                cells.append({
                    "dataset": dataset, "source_model": model, "judge": judge,
                    "broad_problematic": broad, "strict_unsupported": strict,
                    "flags": flags,
                })
    return {
        "design": "paired exact-client legacy versus v4 comparison",
        "judges": sorted(judges_before), "cells": cells,
        "positive_improvement": "lower problematic rate in v4",
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Legacy versus v4 stereotype audit", "",
        "Positive improvement means that the problematic rate decreased in v4.", "",
        "| Dataset | Source | Judge | Legacy | v4 | Improvement | 95% CI | Improved/regressed clients |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["cells"]:
        value = row["broad_problematic"]
        lines.append(
            f"| {row['dataset']} | {row['source_model']} | {row['judge']} | "
            f"{value['legacy_share']:.1%} | {value['v4_share']:.1%} | "
            f"{value['improvement_legacy_minus_v4']:+.1%} | "
            f"[{value['ci_low']:+.1%}, {value['ci_high']:+.1%}] | "
            f"{value['improved_clients']}/{value['regressed_clients']} |"
        )
    lines.extend(["", "## Flag changes", ""])
    for row in result["cells"]:
        changed = [
            (name, values) for name, values in row["flags"].items()
            if values["legacy_share"] or values["v4_share"]
        ]
        if not changed:
            continue
        lines.append(
            f"### {row['dataset']} / {row['source_model']} / {row['judge']}"
        )
        lines.append("")
        for name, values in changed:
            lines.append(
                f"- `{name}`: {values['legacy_share']:.1%} → "
                f"{values['v4_share']:.1%} "
                f"(improvement {values['improvement_legacy_minus_v4']:+.1%})."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "legacy_dir": str(args.legacy_dir), "v4_dir": str(args.v4_dir),
        "output_dir": str(args.output_dir),
    }, indent=2))
    if not args.execute:
        return
    result = compare(
        load_audit(args.legacy_dir, "legacy"),
        load_audit(args.v4_dir, "v4"), seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "comparison.json", result)
    (args.output_dir / "comparison.md").write_text(markdown(result), encoding="utf-8")
    print(f"Saved paired comparison -> {args.output_dir}")


if __name__ == "__main__":
    main()

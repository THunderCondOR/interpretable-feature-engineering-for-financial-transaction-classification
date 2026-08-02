#!/usr/bin/env python3
"""Render and validate representative prompts before any paid API call."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_pipeline import load_prompt_context, load_split
from src.data.aggregator import build_dataset_summary_str
from src.data.entity_ids import canonical_entity_id
from src.data.prompt_locale import assert_english_model_text
from src.experiments.artifacts import atomic_write_json, fingerprint
from src.experiments.config_builder import load_yaml
from src.pipeline.prompt_builder import build_few_shot_str, build_prompts


DATASET_FORBIDDEN = {
    "datafusion_default_2023": (
        "known delinquency", "known debt", "known salary", "known account balance",
    ),
    "cofinfad_operational_fidelity": (
        "* age:", "* gender:", "* location:", "* income bracket:",
        "* occupation:", "* education:", "* marital status:",
        "churn_probability", "customer_lifetime_value",
    ),
}


def _audit_ids(frame: pd.DataFrame) -> list[int | str]:
    profiles = (
        frame.groupby("customer_id", observed=True)
        .agg(label=("label", "first"), count=("amount", "size"), volume=("amount", lambda x: x.abs().sum()))
        .reset_index()
    )
    selected: set[int | str] = set()
    for _, group in profiles.groupby("label", observed=True):
        ranked_count = group.sort_values("count")
        ranked_volume = group.sort_values("volume")
        for ranked in (ranked_count, ranked_volume):
            for position in (0, len(ranked) // 2, max(len(ranked) - 1, 0)):
                # Selecting an entire mixed numeric row with ``iloc`` coerces
                # integer IDs to floats (e.g. 3225 -> "3225.0"), which made
                # the representative audit subset silently empty.
                row_index = ranked.index[position]
                selected.add(canonical_entity_id(ranked.at[row_index, "customer_id"]))
    return sorted(selected, key=str)


def audit(config: dict, output_dir: Path) -> dict:
    train = load_prompt_context(config)
    validation = load_split(config, "val")
    summary = build_dataset_summary_str(train, config)
    few_shot = build_few_shot_str(train, config)
    ids = set(_audit_ids(validation))
    subset = validation[validation["customer_id"].map(canonical_entity_id).isin(ids)]
    prompts = build_prompts(subset, config, summary, few_shot)
    dataset = config["dataset"]["name"]
    forbidden = DATASET_FORBIDDEN.get(dataset, ())
    failures = []
    for record in prompts:
        rendered = record["system_prompt"] + "\n" + record["user_prompt"]
        assert_english_model_text(rendered, context=f"audit prompt {record['customer_id']}")
        lowered = rendered.lower()
        leaked = [term for term in forbidden if term.lower() in lowered]
        if leaked:
            failures.append({"customer_id": record["customer_id"], "forbidden": leaked})
        if str(record["label_name"]) in record["client_stats"]:
            failures.append({"customer_id": record["customer_id"], "target_label_leak": True})
    train_ids = {canonical_entity_id(value) for value in train["customer_id"].unique()}
    validation_ids = {canonical_entity_id(value) for value in validation["customer_id"].unique()}
    if train_ids & validation_ids:
        failures.append({"split_overlap": len(train_ids & validation_ids)})
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "variant": config["experiment"]["variant"],
        "n_audited": len(prompts),
        "audit_ids": [record["customer_id"] for record in prompts],
        "summary_hash": fingerprint(summary),
        "few_shot_hash": fingerprint(few_shot),
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "prompt_audit.json", payload)
    with (output_dir / "rendered_prompts.jsonl").open("w", encoding="utf-8") as stream:
        for record in prompts:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if failures:
        raise ValueError(f"Prompt audit failed: {failures[:5]}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {"mode": "execute" if args.execute else "dry-run", "config": str(args.config), "output_dir": str(args.output_dir)}
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    payload = audit(load_yaml(args.config), args.output_dir)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

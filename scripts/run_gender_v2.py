"""Dry-run-first neutral gender pilot and full-run planner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def stratified_pilot_ids(transactions, n_clients=400, seed=137):
    clients = transactions.groupby("customer_id", sort=False).agg(
        label=("label", "first"), transaction_count=("amount", "size"),
        transaction_volume=("amount", lambda values: values.abs().sum()),
    ).reset_index()
    for column in ("transaction_count", "transaction_volume"):
        clients[f"{column}_quartile"] = pd.qcut(clients[column].rank(method="first"), 4, labels=False, duplicates="drop")
    clients["stratum"] = clients[["label", "transaction_count_quartile", "transaction_volume_quartile"]].astype(str).agg("/".join, axis=1)
    target = min(n_clients, len(clients))
    counts = clients["stratum"].value_counts().sort_index()
    exact = counts / counts.sum() * target
    allocation = np.floor(exact).astype(int)
    for stratum in (exact - allocation).sort_values(ascending=False).index[: target - allocation.sum()]:
        allocation[stratum] += 1
    rng, selected = np.random.default_rng(seed), []
    for stratum, group in clients.groupby("stratum", sort=True):
        take = min(int(allocation.get(stratum, 0)), len(group))
        selected.extend(rng.choice(group["customer_id"].to_numpy(), take, replace=False).tolist())
    return sorted(int(value) for value in selected)


def build_plan(run_id):
    return {
        "run_id": run_id, "dataset": "gender", "pilot_clients": 400, "sampling_seed": 137,
        "qwen_pilot_variants": ["legacy", "neutral_only", "neutral_robust_fewshot", "neutral_robust_zero_shot"],
        "selection": {"split": "val", "primary": "balanced_accuracy", "secondary": ["accuracy", "macro_f1", "parse_success", "social_assertion_audit", "grounding_audit"], "equivalence": "<=1pp and overlapping CI => zero-shot"},
        "full_qwen": ["train", "val", "test", "direct", "claims", "clusters", "cot_ml", "concat_ml"],
        "full_gpt_oss": ["same-seed subset", "test direct", "grounding claims"],
        "reuse": ["majority", "LoRA"], "seeded_reevaluation": ["standard", "handcrafted"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="reviewer-v2")
    parser.add_argument("--validation-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/runs/reviewer-v2"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--until-complete", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.run_id)
    print(json.dumps({"mode": "execute" if args.execute else "dry-run", **plan}, indent=2, ensure_ascii=False))
    if not args.execute:
        return
    if not args.validation_csv:
        raise ValueError("--validation-csv is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = stratified_pilot_ids(pd.read_csv(args.validation_csv))
    (args.output_dir / "gender_pilot_client_ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    (args.output_dir / "gender_v2.queue.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.execute_api:
        if not args.until_complete:
            raise ValueError("--execute-api requires --until-complete")
        raise RuntimeError("Queue prepared; use launch_model_queues.sh for independent model limiters")


if __name__ == "__main__":
    main()

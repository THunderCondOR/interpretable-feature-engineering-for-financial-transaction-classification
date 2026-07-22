"""Prepare claim-level grounding samples.

The script joins the exact client-level summaries supplied to the generator
(`clients_stats_{split}.jsonl`) with extracted atomic claims
(`claims_{split}.jsonl`). It writes JSONL records suitable for LLM judging.
No API calls are made here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs="+", default=["results", "results/gpt_oss_120b"])
    parser.add_argument("--run-names", nargs="+", default=["qwen", "gpt_oss_120b"])
    parser.add_argument("--datasets", nargs="+", default=["gender", "age", "rosbank"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--clients-per-dataset", type=int, default=50)
    parser.add_argument("--claims-per-client", type=int, default=3)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if len(args.run_roots) != len(args.run_names):
        raise ValueError("--run-roots and --run-names must have the same length")

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_records = []

    for run_name, run_root in zip(args.run_names, args.run_roots):
        root = Path(run_root)
        for dataset in args.datasets:
            stats_path = root / dataset / f"clients_stats_{args.split}.jsonl"
            claims_path = root / dataset / f"claims_{args.split}.jsonl"
            summary_path = root / dataset / "summary_stats.txt"
            train_summary = summary_path.read_text(encoding="utf-8")
            stats = {
                int(record["customer_id"]): record
                for record in read_jsonl(stats_path)
            }
            claim_records = [
                record
                for record in read_jsonl(claims_path)
                if record.get("claims") and int(record["customer_id"]) in stats
            ]
            sampled_clients = rng.sample(
                claim_records,
                k=min(args.clients_per_dataset, len(claim_records)),
            )
            for record in sampled_clients:
                cid = int(record["customer_id"])
                claims = [str(c).strip() for c in record.get("claims", []) if str(c).strip()]
                sampled_claims = rng.sample(
                    claims,
                    k=min(args.claims_per_client, len(claims)),
                )
                for claim_idx, claim in enumerate(sampled_claims):
                    out_records.append(
                        {
                            "sample_id": f"{run_name}:{dataset}:{args.split}:{cid}:{claim_idx}",
                            "run_name": run_name,
                            "run_root": str(root),
                            "dataset": dataset,
                            "split": args.split,
                            "customer_id": cid,
                            "client_stats": stats[cid]["client_stats"],
                            "train_reference_summary": train_summary,
                            "evidence_hash": hashlib.sha256(
                                (stats[cid]["client_stats"] + "\n" + train_summary).encode("utf-8")
                            ).hexdigest(),
                            "field_semantics": {
                                "gender": "signed cashflow: negative amounts are outflow",
                                "age": "amount is unsigned transaction value, not income",
                                "rosbank": "direction is defined by operation type",
                            }[dataset],
                            "claim": claim,
                        }
                    )

    rng.shuffle(out_records)
    with open(args.output, "w", encoding="utf-8") as file:
        for record in out_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(out_records)} grounding samples -> {args.output}")


if __name__ == "__main__":
    main()

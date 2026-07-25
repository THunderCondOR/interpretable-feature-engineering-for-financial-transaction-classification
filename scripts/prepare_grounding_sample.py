"""Prepare blinded claim-level grounding samples with verified evidence provenance."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import fingerprint


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def _by_customer(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["customer_id"]): row for row in read_jsonl(path)}


def _claims(record: dict[str, Any]) -> list[dict[str, str]]:
    if record.get("claim_records"):
        return [
            {
                "claim_id": str(row["claim_id"]),
                "claim": str(row.get("original_text", row.get("normalized_text", ""))).strip(),
            }
            for row in record["claim_records"]
            if str(row.get("original_text", row.get("normalized_text", ""))).strip()
        ]
    return [
        {
            "claim_id": fingerprint({
                "customer_id": int(record["customer_id"]),
                "claim": str(claim).strip(),
                "index": index,
            })[:20],
            "claim": str(claim).strip(),
        }
        for index, claim in enumerate(record.get("claims", []))
        if str(claim).strip()
    ]


def _verified_evidence(
    *,
    claim_record: dict[str, Any],
    prompt_record: dict[str, Any] | None,
    stats_record: dict[str, Any],
    train_summary: str,
    allow_unverified_legacy: bool,
) -> tuple[dict[str, Any], str]:
    client_stats = str(stats_record["client_stats"])
    evidence = {
        "client_stats": client_stats,
        "train_reference_summary": train_summary,
    }
    if prompt_record is None:
        if not allow_unverified_legacy:
            raise ValueError("Missing source prompt record")
        return evidence, "legacy_reconstructed"

    expected_client_hash = fingerprint(client_stats)
    expected_summary_hash = fingerprint(train_summary)
    checks = {
        "client_stats_text": prompt_record.get("client_stats") == client_stats,
        "client_stats_hash": prompt_record.get("client_stats_hash") == expected_client_hash,
        "summary_stats_hash": prompt_record.get("summary_stats_hash") == expected_summary_hash,
    }
    source_prompt_hashes = set(claim_record.get("source_prompt_hashes", []))
    if source_prompt_hashes:
        checks["claim_to_prompt"] = prompt_record.get("prompt_hash") in source_prompt_hashes
    elif not allow_unverified_legacy:
        checks["claim_to_prompt"] = False

    if not all(checks.values()):
        if not allow_unverified_legacy:
            failed = sorted(name for name, ok in checks.items() if not ok)
            raise ValueError(f"Grounding evidence provenance failed: {failed}")
        return evidence, "legacy_reconstructed"
    evidence["prompt_hash"] = prompt_record.get("prompt_hash")
    evidence["client_stats_hash"] = expected_client_hash
    evidence["summary_stats_hash"] = expected_summary_hash
    return evidence, "verified_exact_prompt_inputs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs="+", default=["results", "results/gpt_oss_120b"])
    parser.add_argument("--run-names", nargs="+", default=["qwen", "gpt_oss_120b"])
    parser.add_argument(
        "--sources-config",
        type=Path,
        help="YAML registry of exact nested v4 source roots.",
    )
    parser.add_argument("--datasets", nargs="+", default=["gender", "age", "rosbank"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--clients-per-dataset", type=int, default=50)
    parser.add_argument("--claims-per-client", type=int, default=3)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--allow-unverified-legacy", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.sources_config and len(args.run_roots) != len(args.run_names):
        raise ValueError("--run-roots and --run-names must have the same length")

    if not args.execute:
        print(json.dumps({
            "mode": "dry-run",
            "run_roots": args.run_roots,
            "run_names": args.run_names,
            "sources_config": (
                str(args.sources_config) if args.sources_config else None
            ),
            "datasets": args.datasets,
            "split": args.split,
            "clients_per_dataset": args.clients_per_dataset,
            "claims_per_client": args.claims_per_client,
            "allow_unverified_legacy": args.allow_unverified_legacy,
            "output": str(args.output),
        }, indent=2))
        return

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_records = []

    if args.sources_config:
        payload = yaml.safe_load(
            args.sources_config.read_text(encoding="utf-8")
        )
        sources = [
            {
                "run_name": str(row["run_name"]),
                "dataset": str(row["dataset"]),
                "root": Path(row["root"]),
            }
            for row in payload.get("sources", [])
            if str(row.get("dataset")) in args.datasets
        ]
    else:
        sources = [
            {
                "run_name": run_name,
                "dataset": dataset,
                "root": Path(run_root) / dataset,
            }
            for run_name, run_root in zip(args.run_names, args.run_roots)
            for dataset in args.datasets
        ]
    if not sources:
        raise ValueError("No grounding sources configured")

    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sources:
        dataset_root = source["root"]
        dataset = source["dataset"]
        run_name = source["run_name"]
        stats = _by_customer(dataset_root / f"clients_stats_{args.split}.jsonl")
        claims = _by_customer(dataset_root / f"claims_{args.split}.jsonl")
        prompts_path = dataset_root / f"prompts_{args.split}.jsonl"
        prompts = _by_customer(prompts_path) if prompts_path.exists() else {}
        train_summary = (dataset_root / "summary_stats.txt").read_text(
            encoding="utf-8"
        )
        loaded[(dataset, run_name)] = {
            "root": dataset_root,
            "stats": stats,
            "claims": claims,
            "prompts": prompts,
            "train_summary": train_summary,
        }

    for dataset in args.datasets:
        dataset_sources = [
            source for source in sources if source["dataset"] == dataset
        ]
        if not dataset_sources:
            continue
        eligible_sets = []
        for source in dataset_sources:
            item = loaded[(dataset, source["run_name"])]
            eligible_sets.append({
                cid
                for cid, record in item["claims"].items()
                if _claims(record) and cid in item["stats"]
            })
        paired_eligible = sorted(set.intersection(*eligible_sets))
        sampled_ids = rng.sample(
            paired_eligible,
            k=min(args.clients_per_dataset, len(paired_eligible)),
        )
        for source in dataset_sources:
            run_name = source["run_name"]
            item = loaded[(dataset, run_name)]
            stats = item["stats"]
            claims = item["claims"]
            prompts = item["prompts"]
            train_summary = item["train_summary"]
            for cid in sampled_ids:
                evidence, provenance = _verified_evidence(
                    claim_record=claims[cid],
                    prompt_record=prompts.get(cid),
                    stats_record=stats[cid],
                    train_summary=train_summary,
                    allow_unverified_legacy=args.allow_unverified_legacy,
                )
                claim_rows = _claims(claims[cid])
                sampled_claims = rng.sample(
                    claim_rows,
                    k=min(args.claims_per_client, len(claim_rows)),
                )
                for claim_row in sampled_claims:
                    evidence_hash = fingerprint({
                        **evidence,
                        "field_semantics": {
                            "gender": "signed cashflow: negative amounts are outflow",
                            "age": "amount is unsigned transaction value, not income",
                            "rosbank": "direction is defined by operation type",
                        }[dataset],
                    })
                    out_records.append({
                        "sample_id": f"{run_name}:{dataset}:{args.split}:{cid}:{claim_row['claim_id']}",
                        "run_name": run_name,
                        "dataset": dataset,
                        "split": args.split,
                        "customer_id": cid,
                        "client_stats": evidence["client_stats"],
                        "train_reference_summary": evidence["train_reference_summary"],
                        "evidence_hash": evidence_hash,
                        "evidence_provenance": provenance,
                        "source_prompt_hash": evidence.get("prompt_hash"),
                        "field_semantics": {
                            "gender": "signed cashflow: negative amounts are outflow",
                            "age": "amount is unsigned transaction value, not income",
                            "rosbank": "direction is defined by operation type",
                        }[dataset],
                        "claim_id": claim_row["claim_id"],
                        "claim": claim_row["claim"],
                    })

    rng.shuffle(out_records)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        for record in out_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(f"Saved {len(out_records)} grounding samples -> {args.output}")


if __name__ == "__main__":
    main()

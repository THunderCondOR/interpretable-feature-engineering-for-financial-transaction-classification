#!/usr/bin/env python3
"""Build a paired, label-stratified audit sample from exact v4 prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import fingerprint


SAMPLING_PROTOCOL_VERSION = 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def by_customer(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result = {str(row["customer_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate customer_id in {path}")
    return result


def rationale_without_final(text: str) -> str:
    lines = str(text or "").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and re.match(r"^\s*\*{0,2}Final\*{0,2}\s*:", lines[-1], flags=re.I):
        lines.pop()
    rationale = "\n".join(lines).strip()
    if not rationale:
        raise ValueError("Rationale is empty after removing the Final line")
    return rationale


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def materialize_sample(
    *, sources_config: Path, datasets: list[str], split: str,
    clients_per_dataset: int, seed: int,
    reference_manifest: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config = yaml.safe_load(sources_config.read_text(encoding="utf-8")) or {}
    source_rows = [
        row for row in config.get("sources", [])
        if str(row.get("dataset")) in set(datasets)
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for source in source_rows:
        dataset = str(source["dataset"])
        run_name = str(source["run_name"])
        root = Path(source["root"])
        provenance = str(source.get("provenance", "strict_hashed"))
        grouped[dataset].append(source)
        prompt_path = root / f"prompts_{split}.jsonl"
        explanation_path = root / f"explanations_{split}.jsonl"
        summary_path = root / "summary_stats.txt"
        required_paths = [prompt_path, explanation_path, summary_path]
        if provenance == "strict_hashed":
            required_paths.append(root / "manifest.json")
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        prompts = by_customer(prompt_path)
        explanations = by_customer(explanation_path)
        summary = summary_path.read_text(encoding="utf-8")
        loaded[(dataset, run_name)] = {
            "root": root,
            "prompts": prompts,
            "explanations": explanations,
            "summary_hash": fingerprint(summary),
            "condition": str(source.get("condition", "v4")),
            "provenance": provenance,
        }

    reference = (
        json.loads(reference_manifest.read_text(encoding="utf-8"))
        if reference_manifest else None
    )

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for dataset in datasets:
        sources = grouped.get(dataset, [])
        run_names = sorted(str(row["run_name"]) for row in sources)
        if len(run_names) != 2 or len(set(run_names)) != 2:
            raise ValueError(
                f"Expected exactly two source models for {dataset}, got {run_names}"
            )
        eligible_by_model: dict[str, set[str]] = {}
        for run_name in run_names:
            item = loaded[(dataset, run_name)]
            eligible_by_model[run_name] = {
                cid for cid, explanation in item["explanations"].items()
                if cid in item["prompts"]
                and explanation.get("explanation")
                and not explanation.get("error")
                and explanation.get("predicted") is not None
            }
        common = set.intersection(*eligible_by_model.values())
        labels: dict[str, list[str]] = defaultdict(list)
        for cid in common:
            values = {
                str(loaded[(dataset, run_name)]["explanations"][cid]["label"])
                for run_name in run_names
            }
            if len(values) != 1:
                raise ValueError(f"Source-model label mismatch for {dataset}/{cid}")
            labels[next(iter(values))].append(cid)
        if not labels or clients_per_dataset % len(labels):
            raise ValueError(
                f"clients-per-dataset={clients_per_dataset} must be divisible by "
                f"the {len(labels)} labels in {dataset}"
            )
        selected: list[str] = []
        label_counts = {}
        if reference is not None:
            try:
                selected = [
                    str(value)
                    for value in reference["selection"][dataset]["client_ids"]
                ]
            except KeyError as exc:
                raise ValueError(
                    f"Reference manifest has no selection for {dataset}"
                ) from exc
            if len(selected) != clients_per_dataset or len(set(selected)) != len(selected):
                raise ValueError(
                    f"Reference selection for {dataset} must contain exactly "
                    f"{clients_per_dataset} unique clients"
                )
            missing = set(selected) - common
            if missing:
                raise ValueError(
                    f"Legacy sources do not cover {len(missing)} reference clients "
                    f"for {dataset}: {sorted(missing)[:5]}"
                )
            for label, candidates in sorted(labels.items()):
                label_counts[label] = len(set(selected) & set(candidates))
        else:
            per_label = clients_per_dataset // len(labels)
            for label, candidates in sorted(labels.items()):
                if len(candidates) < per_label:
                    raise ValueError(
                        f"Not enough paired clients for {dataset} label {label}: "
                        f"{len(candidates)} < {per_label}"
                    )
                rng = random.Random(f"{seed}:{dataset}:{label}")
                chosen = rng.sample(sorted(candidates), per_label)
                selected.extend(chosen)
                label_counts[label] = len(chosen)
        selection[dataset] = {
            "client_ids": sorted(selected),
            "label_counts": label_counts,
            "source_models": run_names,
        }

        for run_name in run_names:
            item = loaded[(dataset, run_name)]
            for cid in sorted(selected):
                prompt = item["prompts"][cid]
                explanation = item["explanations"][cid]
                client_stats = str(prompt.get("client_stats", ""))
                if item["provenance"] == "strict_hashed":
                    checks = {
                        "prompt_hash": explanation.get("prompt_hash") == prompt.get("prompt_hash"),
                        "client_stats_hash": prompt.get("client_stats_hash")
                        == fingerprint(client_stats),
                        "summary_stats_hash": prompt.get("summary_stats_hash")
                        == item["summary_hash"],
                        "system_prompt": bool(prompt.get("system_prompt")),
                        "user_prompt": bool(prompt.get("user_prompt")),
                    }
                    source_prompt_hash = str(prompt["prompt_hash"])
                    source_client_hash = str(prompt["client_stats_hash"])
                    source_summary_hash = str(prompt["summary_stats_hash"])
                elif item["provenance"] == "legacy_customer_id_join":
                    checks = {
                        "system_prompt": bool(prompt.get("system_prompt")),
                        "user_prompt": bool(prompt.get("user_prompt")),
                        "client_stats": bool(client_stats),
                        "client_in_rendered_prompt": client_stats
                        in str(prompt.get("user_prompt", "")),
                    }
                    source_prompt_hash = fingerprint({
                        "system_prompt": prompt.get("system_prompt"),
                        "user_prompt": prompt.get("user_prompt"),
                        "client_stats": client_stats,
                    })
                    source_client_hash = fingerprint(client_stats)
                    source_summary_hash = item["summary_hash"]
                else:
                    raise ValueError(
                        f"Unknown provenance mode {item['provenance']!r}"
                    )
                if not all(checks.values()):
                    failed = sorted(key for key, ok in checks.items() if not ok)
                    raise ValueError(
                        f"Prompt provenance failed for {dataset}/{run_name}/{cid}: {failed}"
                    )
                rationale = rationale_without_final(explanation["explanation"])
                sample_id = hashlib.sha256(
                    f"{SAMPLING_PROTOCOL_VERSION}:{dataset}:{split}:{cid}:{run_name}:"
                    f"{item['condition']}:{source_prompt_hash}:{fingerprint(rationale)}".encode()
                ).hexdigest()[:24]
                public_rows.append({
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "split": split,
                    "customer_id": cid,
                    "run_name": run_name,
                    "condition": item["condition"],
                    "evidence_provenance": item["provenance"],
                    "source_root": str(item["root"]),
                    "source_prompt_hash": source_prompt_hash,
                    "source_client_stats_hash": source_client_hash,
                    "source_summary_stats_hash": source_summary_hash,
                    "generator_system_prompt": prompt["system_prompt"],
                    "generator_user_prompt": prompt["user_prompt"],
                    "rationale": rationale,
                })
                private_rows.append({
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "customer_id": cid,
                    "run_name": run_name,
                    "condition": item["condition"],
                    "true_label": explanation.get("label"),
                    "true_label_name": explanation.get("label_name"),
                    "predicted": explanation.get("predicted"),
                    "prediction_correct": explanation.get("predicted")
                    == explanation.get("label"),
                })
    public_rows.sort(key=lambda row: row["sample_id"])
    private_rows.sort(key=lambda row: row["sample_id"])
    manifest = {
        "protocol_version": SAMPLING_PROTOCOL_VERSION,
        "sampling_seed": seed,
        "reference_manifest": str(reference_manifest) if reference_manifest else None,
        "split": split,
        "clients_per_dataset": clients_per_dataset,
        "datasets": datasets,
        "n_unique_clients": sum(len(value["client_ids"]) for value in selection.values()),
        "n_rationales": len(public_rows),
        "n_expected_judgments": len(public_rows) * 2,
        "selection": selection,
        "sample_hash": fingerprint(public_rows),
    }
    return public_rows, private_rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources-config", type=Path,
        default=Path("configs/v5/stereotype_audit_sources.yaml"),
    )
    parser.add_argument("--datasets", nargs="+", default=["gender", "age"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--clients-per-dataset", type=int, default=60)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps({
        "mode": "execute" if args.execute else "dry-run",
        "datasets": args.datasets,
        "split": args.split,
        "clients_per_dataset": args.clients_per_dataset,
        "sampling_seed": args.seed,
        "reference_manifest": str(args.reference_manifest)
        if args.reference_manifest else None,
        "output": str(args.output),
    }, indent=2))
    if not args.execute:
        return
    public, private, manifest = materialize_sample(
        sources_config=args.sources_config, datasets=args.datasets,
        split=args.split, clients_per_dataset=args.clients_per_dataset,
        seed=args.seed, reference_manifest=args.reference_manifest,
    )
    _write_jsonl(args.output, public)
    _write_json(args.private_key, private)
    _write_json(args.manifest, manifest)
    print(f"Saved {len(public)} paired rationales -> {args.output}")


if __name__ == "__main__":
    main()

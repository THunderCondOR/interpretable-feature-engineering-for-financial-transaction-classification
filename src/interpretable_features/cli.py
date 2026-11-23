"""Public command-line interface for the reproducibility package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from interpretable_features.evaluation.interpretability_metrics import (
    clustering_agreement,
    grounding_summary,
    surrogate_fidelity,
    validate_judge_records,
)
from interpretable_features.experiments.artifacts import atomic_write_json
from interpretable_features.reporting.analysis_report import (
    build_report,
    load_bundle,
    synthetic_bundle,
)


def _analysis_parser(name: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"ife {name}", description=description)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _run_stability(argv: list[str]) -> None:
    parser = _analysis_parser("stability", "Compare cluster assignments on a common anchor set.")
    parser.add_argument(
        "--axis",
        required=True,
        choices=("generation_seed", "clustering_seed", "threshold", "fixed_k", "coverage"),
    )
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.input)
    required = {"anchor_id", "run", "cluster_id"}
    if not required <= set(frame):
        raise ValueError(f"Input must contain {sorted(required)}")
    pivot = frame.pivot(index="anchor_id", columns="run", values="cluster_id").dropna()
    if len(pivot.columns) < 2 or len(pivot) < 2:
        raise ValueError("Stability requires at least two runs and two shared anchors")
    result = clustering_agreement(
        {str(column): pivot[column].to_numpy() for column in pivot}, args.axis
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


def _run_grounding(argv: list[str]) -> None:
    parser = _analysis_parser("grounding", "Aggregate independent claim-grounding judgments.")
    parser.add_argument("--judges", nargs="+", help="Expected judge names for every sample")
    args = parser.parse_args(argv)
    records = (
        pd.read_json(args.input, lines=args.input.suffix == ".jsonl")
        if args.input.suffix in {".json", ".jsonl"}
        else pd.read_csv(args.input)
    )
    validate_judge_records(records, expected_judges=set(args.judges) if args.judges else None)
    items, summary = grounding_summary(records)
    atomic_write_json(args.output, summary)
    items.to_csv(args.output.with_suffix(".items.csv"), index=False)


def _run_grounding_judge(argv: list[str]) -> None:
    from interpretable_features.analysis.judge_grounding import main

    main(argv)


def _run_fidelity(argv: list[str]) -> None:
    parser = _analysis_parser("fidelity", "Measure claim-model fidelity to a strong teacher.")
    args = parser.parse_args(argv)
    payload = np.load(args.input)
    result = surrogate_fidelity(
        payload["teacher_probabilities"], payload["surrogate_probabilities"], payload["labels"]
    )
    atomic_write_json(args.output, result)


def _run_report(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="ife report", description="Build a self-contained HTML analysis report."
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--run-id", default="synthetic-demo")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if not args.synthetic and args.results_root is None:
        parser.error("--results-root is required unless --synthetic is used")
    bundle = (
        synthetic_bundle()
        if args.synthetic
        else load_bundle(
            args.results_root, args.run_id, allow_missing=args.allow_missing, strict=args.strict
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(bundle, args.run_id), encoding="utf-8")


def _run_prepare(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="ife prepare", description="Prepare a supported dataset and record split provenance."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("gender", "age", "rosbank", "berka", "datafusion_default_2023", "cofinfad"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--test-ids-dir",
        type=Path,
        help="Override the bundled paper split with dataset-specific ID CSVs",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = {
        "dataset": args.dataset,
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "mode": "execute" if args.execute else "dry-run",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if args.dataset in {"gender", "age", "rosbank"}:
        from interpretable_features.data.prepare_data import build_specs, prepare_dataset

        spec = build_specs(args.test_ids_dir, args.output_root)[args.dataset]
        prepare_dataset(spec)
    elif args.dataset == "berka":
        from interpretable_features.data.benchmark_registry import prepare_benchmark

        prepare_benchmark("berka", raw_root=args.raw_root, output_root=args.output_root)
    elif args.dataset == "datafusion_default_2023":
        from interpretable_features.benchmarks.datafusion_default_2023 import prepare

        prepare(args.raw_root, args.output_root)
    else:
        from interpretable_features.benchmarks.cofinfad import prepare

        prepare(args.raw_root, args.output_root)


def _run_pipeline(argv):
    from interpretable_features.runner import main

    main(argv)


def _select_features(argv):
    from interpretable_features.analysis.select_features import main

    main(argv)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "prepare": _run_prepare,
        "run": _run_pipeline,
        "select-features": _select_features,
        "stability": _run_stability,
        "grounding": _run_grounding,
        "fidelity": _run_fidelity,
        "grounding-judge": _run_grounding_judge,
        "report": _run_report,
    }
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: ife {" + ",".join(commands) + "} ...")
        print("Run `ife COMMAND --help` for command-specific options.")
        return
    command, rest = argv[0], argv[1:]
    if command not in commands:
        raise SystemExit(f"Unknown command: {command}")
    commands[command](rest)

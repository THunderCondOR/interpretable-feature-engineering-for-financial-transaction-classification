"""Run dataset experiments from a single split-aware entry point.

Examples:
    python run_pipeline.py --config configs/gender.yaml --steps stats,prompts,cot,llm_eval --splits test
    python run_pipeline.py --config configs/age.yaml --steps prompts,cot,claims --splits train,val,test
    python run_pipeline.py --config configs/rosbank.yaml --steps cot_features,ml --experiments handcrafted,cot,concat
    python run_pipeline.py --config configs/gender.yaml --steps lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.data.aggregator import build_all_client_stats, build_dataset_summary_str
from src.data.loader import add_features, load_dataset
from src.models.lora_trainer import train as lora_train
from src.models.ml_baseline import run_ml_baseline
from src.pipeline.claims_extractor import run_claims_extraction
from src.pipeline.cot_features import build_cot_features
from src.pipeline.explanation_gen import run_explanation_generation
from src.pipeline.llm_eval import evaluate_llm_predictions
from src.pipeline.prompt_builder import build_few_shot_str, build_prompts

SPLITS = ("train", "val", "test")
DEFAULT_STEPS = ("stats", "prompts", "cot", "llm_eval", "claims", "cot_features", "ml")
STEPS = DEFAULT_STEPS + ("lora",)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if "label_names" in config.get("dataset", {}):
        config["dataset"]["label_names"] = {str(k): v for k, v in config["dataset"]["label_names"].items()}
    return config


def split_output_path(config: dict, key: str, split: str) -> Path:
    out_dir = Path(config["output"]["base_dir"])
    base = Path(config["output"][key])
    return out_dir / f"{base.stem}_{split}{base.suffix}"


def load_split(config: dict, split: str) -> pd.DataFrame:
    return add_features(load_dataset(config, split))


def run_stats(config: dict, splits: list[str]) -> None:
    out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(config, "train")
    summary = build_dataset_summary_str(train_df, config)
    summary_path = out_dir / config["output"]["summary_stats"]
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Saved train summary -> {summary_path}")

    for split in splits:
        df = load_split(config, split)
        records = build_all_client_stats(df, config)
        path = split_output_path(config, "clients_stats", split)
        with open(path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {split} client stats: {len(records)} clients -> {path}")


def run_prompts(config: dict, splits: list[str]) -> None:
    out_dir = Path(config["output"]["base_dir"])
    summary_path = out_dir / config["output"]["summary_stats"]
    summary = summary_path.read_text(encoding="utf-8")
    train_df = load_split(config, "train")
    few_shot = build_few_shot_str(train_df, config)

    for split in splits:
        df = load_split(config, split)
        records = build_prompts(df, config, summary, few_shot)
        path = split_output_path(config, "prompts", split)
        with open(path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {split} prompts: {len(records)} clients -> {path}")


def run_cot(config: dict, splits: list[str]) -> None:
    for split in splits:
        run_explanation_generation(config, split=split)


def run_llm_eval(config: dict, splits: list[str]) -> None:
    for split in splits:
        evaluate_llm_predictions(config, split)


def run_claims(config: dict, splits: list[str]) -> None:
    for split in splits:
        run_claims_extraction(config, split=split)


def run_cot_features(config: dict, splits: list[str]) -> None:
    build_cot_features(config)


def run_ml(config: dict, splits: list[str], experiments: list[str]) -> None:
    run_ml_baseline(config, experiments=experiments)


def run_lora(config: dict, splits: list[str]) -> None:
    lora_train(config)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transaction classification experiments.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    parser.add_argument("--steps", default="default", help="Comma-separated steps, 'default', or 'all'.")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split list for split-aware steps.")
    parser.add_argument(
        "--experiments",
        default="handcrafted,cot,concat",
        help="Comma-separated ML feature sets: handcrafted,cot,concat.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    splits = parse_csv(args.splits)
    experiments = parse_csv(args.experiments)

    if args.steps == "default":
        steps = list(DEFAULT_STEPS)
    elif args.steps == "all":
        steps = list(STEPS)
    else:
        steps = parse_csv(args.steps)

    unknown_steps = [step for step in steps if step not in STEPS]
    unknown_splits = [split for split in splits if split not in SPLITS]
    unknown_experiments = [exp for exp in experiments if exp not in {"handcrafted", "cot", "concat"}]
    if unknown_steps or unknown_splits or unknown_experiments:
        print(f"Unknown steps: {unknown_steps}", file=sys.stderr)
        print(f"Unknown splits: {unknown_splits}", file=sys.stderr)
        print(f"Unknown experiments: {unknown_experiments}", file=sys.stderr)
        sys.exit(1)

    step_fns = {
        "stats": lambda: run_stats(config, splits),
        "prompts": lambda: run_prompts(config, splits),
        "cot": lambda: run_cot(config, splits),
        "llm_eval": lambda: run_llm_eval(config, splits),
        "claims": lambda: run_claims(config, splits),
        "cot_features": lambda: run_cot_features(config, splits),
        "ml": lambda: run_ml(config, splits, experiments),
        "lora": lambda: run_lora(config, splits),
    }

    for step in steps:
        print(f"\n{'=' * 72}\nStep: {step}\n{'=' * 72}")
        step_fns[step]()

    print("\nDone.")


if __name__ == "__main__":
    main()

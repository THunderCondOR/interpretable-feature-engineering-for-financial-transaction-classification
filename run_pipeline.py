"""Run dataset experiments from a single split-aware entry point.

Examples:
    python run_pipeline.py --config configs/gender.yaml --steps stats,prompts,cot,llm_eval --splits test
    python run_pipeline.py --config configs/gender.yaml --steps cot,llm_eval --splits test
    python run_pipeline.py --config configs/gender.yaml --steps cot --splits test --no-resume-cot
    python run_pipeline.py --config configs/gender.yaml --steps lora
    python run_pipeline.py --config configs/gender.yaml --steps lora --lora-models qwen_1_5b,qwen_7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
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
STEPS = DEFAULT_STEPS + ("majority", "lora")


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


def run_cot(config: dict, splits: list[str], *, resume_cot: bool) -> None:
    for split in splits:
        run_explanation_generation(config, split=split, resume=resume_cot)


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


def run_lora(config: dict, splits: list[str], lora_models: list[str] | None) -> None:
    lora_train(config, model_names=lora_models)


def run_majority(config: dict, splits: list[str]) -> None:
    train = load_split(config, "train").drop_duplicates("customer_id")
    majority_label = int(train["label"].value_counts().idxmax())
    results = {
        "majority_label": majority_label,
        "majority_label_name": config["dataset"]["label_names"][str(majority_label)],
        "splits": {},
    }
    for split in splits:
        clients = load_split(config, split).drop_duplicates("customer_id")
        predictions = np.full(len(clients), majority_label)
        from src.models.ml_baseline import evaluate
        results["splits"][split] = evaluate(NoneModel(predictions), predictions, clients["label"].to_numpy())

    out_path = Path(config["output"]["base_dir"]) / "metrics_majority.json"
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(out_path)
    print(f"Saved majority metrics -> {out_path}")


class NoneModel:
    """Minimal constant predictor adapter for the shared evaluator."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, _):
        return self.predictions


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transaction classification experiments.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    parser.add_argument("--model", default=None, help="Override llm.default_model from config.")
    parser.add_argument("--output-base-dir", default=None, help="Override output.base_dir from config.")
    parser.add_argument("--max-concurrent", type=int, default=None, help="Override LLM concurrency.")
    parser.add_argument(
        "--rate-limit-fallback-concurrent",
        type=int,
        default=None,
        help="Override LLM concurrency after a rate-limit batch.",
    )
    parser.add_argument(
        "--rate-limit-recovery-batches",
        type=int,
        default=None,
        help="Clean fallback batches before retrying the configured LLM concurrency.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override LLM batch size.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override explanation token budget.")
    parser.add_argument("--claims-max-tokens", type=int, default=None, help="Override claim extraction token budget.")
    parser.add_argument("--steps", default="default", help="Comma-separated steps, 'default', or 'all'.")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split list for split-aware steps.")
    parser.add_argument(
        "--experiments",
        default="standard,handcrafted,cot,concat",
        help="Comma-separated ML feature sets: standard,handcrafted,cot,concat.",
    )
    parser.add_argument(
        "--resume-cot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse successful existing explanations and resend only failed or missing requests "
            "(default: enabled; use --no-resume-cot to force a full rerun)."
        ),
    )
    parser.add_argument(
        "--lora-models",
        default=None,
        help="Comma-separated LoRA run names or model ids. Defaults to all runs listed in config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["llm"]["default_model"] = args.model
    if args.output_base_dir:
        config["output"]["base_dir"] = args.output_base_dir
    if args.max_concurrent is not None:
        config["llm"]["max_concurrent"] = args.max_concurrent
    if args.rate_limit_fallback_concurrent is not None:
        config["llm"]["rate_limit_fallback_concurrent"] = args.rate_limit_fallback_concurrent
    if args.rate_limit_recovery_batches is not None:
        config["llm"]["rate_limit_recovery_batches"] = args.rate_limit_recovery_batches
    if args.batch_size is not None:
        config["llm"]["batch_size"] = args.batch_size
    if args.max_tokens is not None:
        config["llm"]["max_tokens"] = args.max_tokens
    if args.claims_max_tokens is not None:
        config.setdefault("pipeline", {})["claims_max_tokens"] = args.claims_max_tokens
    splits = parse_csv(args.splits)
    experiments = parse_csv(args.experiments)
    lora_models = parse_csv(args.lora_models) if args.lora_models else None

    if args.steps == "default":
        steps = list(DEFAULT_STEPS)
    elif args.steps == "all":
        steps = list(STEPS)
    else:
        steps = parse_csv(args.steps)

    unknown_steps = [step for step in steps if step not in STEPS]
    unknown_splits = [split for split in splits if split not in SPLITS]
    unknown_experiments = [exp for exp in experiments if exp not in {"standard", "handcrafted", "cot", "concat"}]
    if unknown_steps or unknown_splits or unknown_experiments:
        print(f"Unknown steps: {unknown_steps}", file=sys.stderr)
        print(f"Unknown splits: {unknown_splits}", file=sys.stderr)
        print(f"Unknown experiments: {unknown_experiments}", file=sys.stderr)
        sys.exit(1)

    step_fns = {
        "stats": lambda: run_stats(config, splits),
        "prompts": lambda: run_prompts(config, splits),
        "cot": lambda: run_cot(config, splits, resume_cot=args.resume_cot),
        "llm_eval": lambda: run_llm_eval(config, splits),
        "claims": lambda: run_claims(config, splits),
        "cot_features": lambda: run_cot_features(config, splits),
        "ml": lambda: run_ml(config, splits, experiments),
        "majority": lambda: run_majority(config, splits),
        "lora": lambda: run_lora(config, splits, lora_models),
    }

    for step in steps:
        print(f"\n{'=' * 72}\nStep: {step}\n{'=' * 72}")
        step_fns[step]()

    print("\nDone.")


if __name__ == "__main__":
    main()

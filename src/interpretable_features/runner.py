"""Run the complete experiment from a single split-aware entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from interpretable_features.data.aggregator import build_all_client_stats, build_dataset_summary_str
from interpretable_features.data.loader import add_features, load_dataset
from interpretable_features.data.profiles import export_robust_statistics, robust_statistics_payload
from interpretable_features.experiments.artifacts import ensure_run_manifest
from interpretable_features.experiments.config_builder import deep_merge, load_yaml
from interpretable_features.experiments.events import append_event
from interpretable_features.models.ml_baseline import EXPERIMENTS, evaluate, run_ml_baseline
from interpretable_features.pipeline.claims_extractor import run_claims_extraction
from interpretable_features.pipeline.cot_features import build_cot_features, split_output_path
from interpretable_features.pipeline.explanation_gen import run_explanation_generation
from interpretable_features.pipeline.llm_eval import evaluate_llm_predictions
from interpretable_features.pipeline.prompt_builder import (
    build_few_shot_str,
    build_prompts,
    prompt_length_telemetry,
)

SPLITS = ("train", "val", "test")
DEFAULT_STEPS = ("stats", "prompts", "cot", "llm_eval", "claims", "cot_features", "ml")
STEPS = DEFAULT_STEPS + ("majority", "lora")
API_STEPS = frozenset({"cot", "claims"})


def prompt_context_split(config: dict) -> str:
    """Return the only split allowed to fit label-conditioned prompt context."""
    split = str(config.get("pipeline", {}).get("prompt_context_split", "train"))
    if split != "train":
        raise ValueError(
            "Dataset summaries and few-shot demonstrations must use train only; "
            f"received prompt_context_split={split!r}"
        )
    return split


def pipeline_event(config: dict, event: str, stage: str, **payload) -> None:
    events_path = config.get("llm", {}).get("events_path")
    if not events_path:
        events_path = str(Path(config["output"]["base_dir"]) / "events.jsonl")
    append_event(
        events_path,
        event=event,
        state="completed" if event == "stage_completed" else "running",
        run_id=config.get("experiment", {}).get("run_id"),
        dataset=config.get("dataset", {}).get("name", "unknown"),
        model=config.get("experiment", {}).get(
            "model_slug", config.get("llm", {}).get("default_model", "unknown")
        ),
        stage=stage,
        **payload,
    )


def load_config(path: str, model_path: str | None = None) -> dict:
    dataset_path = Path(path)
    config = load_yaml(dataset_path)
    if defaults := config.pop("defaults", None):
        defaults_path = dataset_path.parent / defaults
        common = load_yaml(defaults_path)
        default_model = defaults_path.parent / common.pop("default_model_config")
        config = deep_merge(common, config)
        model_path = model_path or str(default_model)
    if model_path:
        config = deep_merge(config, load_yaml(model_path))
    if "label_names" in config.get("dataset", {}):
        names = config["dataset"]["label_names"]
        config["dataset"]["label_names"] = {str(k): v for k, v in names.items()}
    return config


def load_split(
    config: dict,
    split: str,
    *,
    apply_client_filter: bool = True,
) -> pd.DataFrame:
    return add_features(
        load_dataset(
            config,
            split,
            apply_client_filter=apply_client_filter,
        ),
        minimal=(config.get("pipeline", {}).get("event_features") == "minimal"),
    )


def load_prompt_context(config: dict) -> pd.DataFrame:
    """Load the complete train split used for summaries and demonstrations."""
    population = str(
        config.get("pipeline", {}).get("prompt_context_population", "full_train_split")
    )
    if population != "full_train_split":
        raise ValueError(
            "Prompt context must use the complete train split; received "
            f"prompt_context_population={population!r}"
        )
    return load_split(
        config,
        prompt_context_split(config),
        apply_client_filter=bool(
            config.get("pipeline", {}).get("prompt_context_apply_client_filter", False)
        ),
    )


def run_stats(config: dict, splits: list[str]) -> None:
    out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_prompt_context(config)
    summary = build_dataset_summary_str(train_df, config)
    summary_path = out_dir / config["output"]["summary_stats"]
    summary_path.write_text(summary, encoding="utf-8")
    export_robust_statistics(
        robust_statistics_payload(train_df[train_df["label"] >= 0], config),
        out_dir / "statistics",
    )
    print(
        f"Saved train summary from "
        f"{train_df['customer_id'].nunique()} reference clients -> "
        f"{summary_path}"
    )

    for split in splits:
        df = load_split(config, split)
        records = build_all_client_stats(df, config)
        path = split_output_path(config, "clients_stats", split)
        with open(path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {split} client stats: {len(records)} clients -> {path}")
    pipeline_event(
        config,
        "stage_completed",
        "stats",
        completed=len(splits),
        expected=len(splits),
        splits=splits,
    )


def run_prompts(config: dict, splits: list[str]) -> None:
    out_dir = Path(config["output"]["base_dir"])
    summary_path = out_dir / config["output"]["summary_stats"]
    summary = summary_path.read_text(encoding="utf-8")
    train_df = load_prompt_context(config)
    few_shot = build_few_shot_str(train_df, config)

    for split in splits:
        df = load_split(config, split)
        records = build_prompts(df, config, summary, few_shot)
        path = split_output_path(config, "prompts", split)
        with open(path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        telemetry_path = out_dir / f"prompt_length_telemetry_{split}.json"
        temporary = telemetry_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                prompt_length_telemetry(records),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(telemetry_path)
        print(f"Saved {split} prompts: {len(records)} clients -> {path}")
    pipeline_event(
        config,
        "stage_completed",
        "prompts",
        completed=len(splits),
        expected=len(splits),
        splits=splits,
    )


def run_cot(config: dict, splits: list[str], *, resume_cot: bool) -> None:
    for split in splits:
        run_explanation_generation(config, split=split, resume=resume_cot)
        pipeline_event(
            config,
            "stage_completed",
            "explanations",
            split=split,
        )


def run_llm_eval(config: dict, splits: list[str]) -> None:
    for split in splits:
        evaluate_llm_predictions(config, split)
    pipeline_event(
        config,
        "stage_completed",
        "direct_eval",
        completed=len(splits),
        expected=len(splits),
        splits=splits,
    )


def run_claims(config: dict, splits: list[str]) -> None:
    for split in splits:
        run_claims_extraction(config, split=split)
        pipeline_event(
            config,
            "stage_completed",
            "claims",
            split=split,
        )


def run_cot_features(config: dict, splits: list[str]) -> None:
    build_cot_features(config)


def run_ml(config: dict, splits: list[str], experiments: list[str]) -> None:
    run_ml_baseline(config, experiments=experiments)


def run_lora(config: dict, splits: list[str], lora_models: list[str] | None) -> None:
    from interpretable_features.models.lora_trainer import train as lora_train

    lora_train(config, model_names=lora_models)


def run_majority(config: dict, splits: list[str]) -> None:
    from sklearn.dummy import DummyClassifier

    train = load_dataset(config, "train").drop_duplicates("customer_id")
    model = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), train["label"])
    majority_label = int(model.predict([[0]])[0])
    results = {
        "majority_label": majority_label,
        "majority_label_name": config["dataset"]["label_names"][str(majority_label)],
        "splits": {},
    }
    for split in splits:
        clients = load_dataset(config, split).drop_duplicates("customer_id")
        results["splits"][split] = evaluate(
            model, np.zeros((len(clients), 1)), clients["label"].to_numpy()
        )

    out_path = Path(config["output"]["base_dir"]) / "metrics_majority.json"
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(out_path)
    print(f"Saved majority metrics -> {out_path}")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run transaction classification experiments.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    parser.add_argument(
        "--model-config",
        default=None,
        help="Optional model profile merged into the dataset config.",
    )
    parser.add_argument("--model", default=None, help="Override llm.default_model from config.")
    parser.add_argument(
        "--output-base-dir", default=None, help="Override output.base_dir from config."
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=None, help="Override LLM concurrency."
    )
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
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Override explanation token budget."
    )
    parser.add_argument(
        "--claims-max-tokens",
        type=int,
        default=None,
        help="Override claim extraction token budget.",
    )
    parser.add_argument(
        "--steps", default="default", help="Comma-separated steps, 'default', or 'all'."
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated split list for split-aware steps.",
    )
    parser.add_argument(
        "--experiments",
        default="standard,handcrafted,cot,concat",
        help="Comma-separated feature sets: " + ", ".join(sorted(EXPERIMENTS)),
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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the printed plan. Without this flag the runner is read-only.",
    )
    parser.add_argument(
        "--until-complete",
        action="store_true",
        help="Required for stages that issue LLM API requests.",
    )
    parser.add_argument("--fold", type=int, choices=range(5), help="Berka outer split (default 0).")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.model_config)
    if config["dataset"]["name"] == "berka" and "client_ids_by_split" in config["dataset"]:
        fold = args.fold or 0
        config["dataset"]["client_ids_by_split"] = {
            split: path.replace("{fold}", str(fold))
            for split, path in config["dataset"]["client_ids_by_split"].items()
        }
        config["output"]["base_dir"] = str(Path(config["output"]["base_dir"]) / f"fold_{fold}")
        config["experiment"]["fold"] = fold
    elif args.fold is not None:
        parser.error("--fold requires the fold-based Berka configuration")
    if args.model_config:
        model_slug = Path(args.model_config).stem
        config.setdefault("experiment", {})["model_slug"] = model_slug
        run_id = str(config["experiment"].get("run_id", config["dataset"]["name"]))
        if not run_id.endswith(f"-{model_slug}"):
            config["experiment"]["run_id"] = f"{run_id}-{model_slug}"
        config["output"]["base_dir"] = str(Path(config["output"]["base_dir"]) / model_slug)
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
        config.setdefault("claims_generation", {})["max_tokens"] = args.claims_max_tokens

    splits = parse_csv(args.splits)
    experiments = parse_csv(args.experiments)
    lora_models = parse_csv(args.lora_models) if args.lora_models else None

    if args.steps == "default":
        steps = list(DEFAULT_STEPS)
        if config["dataset"]["name"] == "cofinfad_operational_fidelity":
            steps.remove("ml")
    elif args.steps == "all":
        steps = list(STEPS)
    else:
        steps = parse_csv(args.steps)

    unknown_steps = [step for step in steps if step not in STEPS]
    unknown_splits = [split for split in splits if split not in SPLITS]
    unknown_experiments = [exp for exp in experiments if exp not in EXPERIMENTS]
    if not steps or not splits or ("ml" in steps and not experiments):
        parser.error("Select at least one step, split and requested ML feature set")
    if unknown_steps or unknown_splits or unknown_experiments:
        print(f"Unknown steps: {unknown_steps}", file=sys.stderr)
        print(f"Unknown splits: {unknown_splits}", file=sys.stderr)
        print(f"Unknown experiments: {unknown_experiments}", file=sys.stderr)
        sys.exit(1)

    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "config": str(args.config),
        "output_base_dir": config["output"]["base_dir"],
        "steps": steps,
        "splits": splits,
        "experiments": experiments,
        "api_steps": sorted(set(steps) & API_STEPS),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.execute:
        return
    if set(steps) & API_STEPS and not args.until_complete:
        raise ValueError("API stages require --until-complete")
    if not config.get("experiment"):
        raise RuntimeError("Refusing to write without a versioned experiment block.")

    manifest_path = ensure_run_manifest(config, repo_root=Path.cwd())
    print(f"Experiment manifest -> {manifest_path}")

    if args.until_complete:
        config.setdefault("execution", {})["until_complete"] = True

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

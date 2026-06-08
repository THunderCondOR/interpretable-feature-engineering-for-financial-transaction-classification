"""Sanity checks for configs, prompts, prepared data, and result files.

Examples:
    python check_pipeline.py --dataset all
    python check_pipeline.py --dataset gender --stage data
    python check_pipeline.py --dataset rosbank --stage cot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

DATASETS = ("gender", "age", "rosbank")
STAGES = ("config", "data", "llm", "cot")


def load_config(dataset: str) -> dict:
    path = Path("configs") / f"{dataset}.yaml"
    assert path.exists(), f"Missing config: {path}"
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def check_prompt_file(base_dir: str, relative_path: str) -> None:
    path = Path(base_dir) / relative_path
    assert path.exists(), f"Missing prompt file: {path}"
    assert path.read_text(encoding="utf-8").strip(), f"Empty prompt file: {path}"


def check_config(dataset: str) -> None:
    config = load_config(dataset)
    for section in ["dataset", "llm", "prompts", "lora", "pipeline", "optuna", "output"]:
        assert section in config, f"{dataset}: missing section {section}"

    dataset_cfg = config["dataset"]
    assert dataset_cfg["name"] == dataset
    for key in ["columns", "label_names", "num_labels", "splits"]:
        assert key in dataset_cfg, f"{dataset}: missing dataset.{key}"
    for split in ["train", "val", "test"]:
        assert split in dataset_cfg["splits"], f"{dataset}: missing split path {split}"

    prompt_cfg = config["prompts"]
    for key in ["system", "user", "claims_system", "claims_user"]:
        check_prompt_file(prompt_cfg["base_dir"], prompt_cfg[key])

    output_cfg = config["output"]
    for key in ["base_dir", "summary_stats", "clients_stats", "prompts", "explanations", "claims", "metrics"]:
        assert key in output_cfg, f"{dataset}: missing output.{key}"

    print(f"[OK] {dataset}: config and prompt files")


def ids_from_csv(path: Path, column: str) -> set[int]:
    return set(pd.read_csv(path, usecols=[column])[column].astype("int64"))


def check_data(dataset: str) -> None:
    config = load_config(dataset)
    split_paths = {split: Path(path) for split, path in config["dataset"]["splits"].items()}
    for split, path in split_paths.items():
        assert path.exists(), f"{dataset}: missing prepared split {split}: {path}"
        df = pd.read_csv(path, nrows=1000)
        required = {"customer_id", "tr_datetime", "amount", "mcc_code_desc", "label"}
        assert required <= set(df.columns), f"{dataset}/{split}: missing columns {required - set(df.columns)}"

    train_ids = ids_from_csv(split_paths["train"], "customer_id")
    val_ids = ids_from_csv(split_paths["val"], "customer_id")
    test_ids = ids_from_csv(split_paths["test"], "customer_id")
    assert not (train_ids & val_ids), f"{dataset}: train/val overlap"
    assert not (train_ids & test_ids), f"{dataset}: train/test overlap"
    assert not (val_ids & test_ids), f"{dataset}: val/test overlap"

    report_path = split_paths["train"].parent / "split_report.json"
    summary_path = split_paths["train"].parent / "split_summary.txt"
    assert report_path.exists(), f"{dataset}: missing {report_path}"
    assert summary_path.exists(), f"{dataset}: missing {summary_path}"
    with open(report_path, encoding="utf-8") as file:
        report = json.load(file)
    assert report["dataset"] == dataset

    print(f"[OK] {dataset}: prepared data splits")


def split_result_path(config: dict, key: str, split: str) -> Path:
    base = Path(config["output"][key])
    return Path(config["output"]["base_dir"]) / f"{base.stem}_{split}{base.suffix}"


def check_llm_outputs(dataset: str) -> None:
    config = load_config(dataset)
    out_dir = Path(config["output"]["base_dir"])
    assert (out_dir / config["output"]["summary_stats"]).exists(), f"{dataset}: missing summary stats"
    for split in ["train", "val", "test"]:
        for key in ["clients_stats", "prompts", "explanations"]:
            path = split_result_path(config, key, split)
            assert path.exists(), f"{dataset}: missing {path}"
    print(f"[OK] {dataset}: LLM-stage outputs")


def check_cot_outputs(dataset: str) -> None:
    config = load_config(dataset)
    out_dir = Path(config["output"]["base_dir"])
    for split in ["train", "val", "test"]:
        claims_path = split_result_path(config, "claims", split)
        features_path = out_dir / f"cot_features_{split}.parquet"
        assert claims_path.exists(), f"{dataset}: missing {claims_path}"
        assert features_path.exists(), f"{dataset}: missing {features_path}"
        df = pd.read_parquet(features_path)
        assert {"customer_id", "label"} <= set(df.columns), f"{dataset}: invalid {features_path}"
    assert (out_dir / "cot_clusters.json").exists(), f"{dataset}: missing cot_clusters.json"
    print(f"[OK] {dataset}: CoT-stage outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check pipeline files and outputs.")
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--stage", choices=["all", *STAGES], default="config")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    stages = STAGES if args.stage == "all" else (args.stage,)
    checks = {
        "config": check_config,
        "data": check_data,
        "llm": check_llm_outputs,
        "cot": check_cot_outputs,
    }
    for dataset in datasets:
        for stage in stages:
            checks[stage](dataset)


if __name__ == "__main__":
    main()

"""Assemble every Table 1 row into one machine-readable result file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("gender", "age", "rosbank")
ARTICLE_ACCURACY = {
    "majority": {"gender": 0.575, "age": 0.260, "rosbank": 0.553},
    "gpt_oss_120b": {"gender": 0.628, "age": 0.338, "rosbank": 0.521},
    "deepseek_r1": {"gender": 0.650, "age": 0.311, "rosbank": 0.526},
    "lora_1_5b": {"gender": 0.690, "age": 0.533, "rosbank": 0.653},
    "lora_7b": {"gender": 0.675, "age": 0.534, "rosbank": 0.661},
    "lora_14b": {"gender": 0.677, "age": 0.540, "rosbank": 0.675},
    "cot": {"gender": 0.683, "age": 0.413, "rosbank": 0.605},
    "standard": {"gender": 0.774, "age": 0.642, "rosbank": 0.743},
    "handcrafted": {"gender": 0.793, "age": 0.600, "rosbank": 0.744},
    "concat": {"gender": 0.786, "age": 0.601, "rosbank": 0.739},
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def metric_subset(metrics: dict) -> dict:
    keys = ("n", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "roc_auc")
    return {key: metrics[key] for key in keys if key in metrics}


def lora_metric_subset(metrics: dict) -> dict:
    result = {}
    for key in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "roc_auc"):
        source_key = f"test_{key}"
        if source_key in metrics:
            result[key] = metrics[source_key]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {"article_accuracy": ARTICLE_ACCURACY, "datasets": {}}
    for dataset in DATASETS:
        run_dir = args.run_root / dataset
        reference_dir = args.reference_root / dataset
        ml = read_json(run_dir / "ml_metrics.json")
        majority = read_json(run_dir / "metrics_majority.json")
        direct = read_json(run_dir / "llm_metrics_test.json")
        lora = read_json(reference_dir / "lora_metrics.json")

        rows = {
            "majority": metric_subset(majority["splits"]["test"]),
            "gpt_oss_120b": metric_subset(direct),
            "lora_1_5b": lora_metric_subset(lora["deepseek_qwen_1_5b"]),
            "lora_7b": lora_metric_subset(lora["deepseek_qwen_7b"]),
            "lora_14b": lora_metric_subset(lora["deepseek_qwen_14b"]),
        }
        for feature_set in ("cot", "standard", "handcrafted", "concat"):
            rows[feature_set] = metric_subset(ml[feature_set]["xgboost"]["test"])

        # No compatible current DeepSeek-R1 API artifact exists in the repository.
        rows["deepseek_r1"] = {
            "status": "article_reference_only",
            "accuracy": ARTICLE_ACCURACY["deepseek_r1"][dataset],
        }
        result["datasets"][dataset] = rows

    for dataset, rows in result["datasets"].items():
        for method, metrics in rows.items():
            current = metrics.get("accuracy")
            article = ARTICLE_ACCURACY.get(method, {}).get(dataset)
            if current is not None and article is not None:
                metrics["article_accuracy"] = article
                metrics["accuracy_diff"] = current - article

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temp_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(args.output)
    print(f"Saved complete Table 1 summary -> {args.output}")


if __name__ == "__main__":
    main()

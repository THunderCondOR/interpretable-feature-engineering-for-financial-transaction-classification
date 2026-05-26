import json
import time
import gc
from pathlib import Path

import yaml
import torch

from src.models.lora_trainer import train


RUNS = [
    {
        "label": "rosbank_1.5B",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "batch_size": 16,
        "gradient_accumulation_steps": 2,
    },
    {
        "label": "rosbank_7B",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
    },
    {
        "label": "rosbank_14B",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
    },
]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["dataset"]["label_names"] = {
        str(k): v for k, v in config["dataset"]["label_names"].items()
    }
    return config


def run_one(run: dict) -> dict:
    config = load_config("configs/rosbank.yaml")

    config["lora"]["model_name"] = run["model_name"]
    config["lora"]["batch_size"] = run["batch_size"]
    config["lora"]["gradient_accumulation_steps"] = run["gradient_accumulation_steps"]
    config["lora"]["max_length"] = 1024
    config["lora"]["load_in_4bit"] = True

    config["output"]["base_dir"] = f"results/{run['label']}"
    config["output"]["metrics"] = "metrics.json"

    Path(config["output"]["base_dir"]).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train(config)
    elapsed_min = round((time.time() - t0) / 60, 2)

    metrics_path = Path(config["output"]["base_dir"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    metrics["elapsed_minutes"] = elapsed_min
    metrics["model_name"] = run["model_name"]
    metrics["batch_size"] = run["batch_size"]
    metrics["gradient_accumulation_steps"] = run["gradient_accumulation_steps"]

    return metrics


def main():
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    summary = {}

    for run in RUNS:
        label = run["label"]

        print("\n" + "=" * 80)
        print(f"Starting {label}")
        print(f"Model: {run['model_name']}")
        print(f"Batch size: {run['batch_size']}")
        print(f"Gradient accumulation: {run['gradient_accumulation_steps']}")
        print("=" * 80)

        try:
            metrics = run_one(run)
            summary[label] = {"status": "ok", **metrics}
            print(f"Finished {label}")
            print(json.dumps(metrics, indent=2, ensure_ascii=False))

        except Exception as e:
            summary[label] = {"status": "error", "error": repr(e)}
            print(f"ERROR in {label}: {repr(e)}")

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            Path("results/rosbank_lora_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    print("\nFinal summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
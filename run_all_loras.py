"""
run_all_loras.py

Последовательно обучает LoRA модели для всех датасетов.

Запуск:
    PYTHONPATH=. python run_all_loras.py
    PYTHONPATH=. python run_all_loras.py  # только активные (не закомментированные) RUNS
"""

import json
import time
import yaml
from pathlib import Path
from tqdm import tqdm

from src.models.lora_trainer import train


# ---------------------------------------------------------------------------
# Конфигурации запусков
# Добавь нужные, закомментируй ненужные
# ---------------------------------------------------------------------------

RUNS = [
    # --- Gender ---
    {"dataset": "gender", "config": "configs/gender.yaml", "model_size": "1.5B",
     "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
    # {"dataset": "gender", "config": "configs/gender.yaml", "model_size": "7B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    # {"dataset": "gender", "config": "configs/gender.yaml", "model_size": "14B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"},

    # --- Age ---
    {"dataset": "age", "config": "configs/age.yaml", "model_size": "1.5B",
     "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
    # {"dataset": "age", "config": "configs/age.yaml", "model_size": "7B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    # {"dataset": "age", "config": "configs/age.yaml", "model_size": "14B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"},

    # --- Rosbank ---
    {"dataset": "rosbank", "config": "configs/rosbank.yaml", "model_size": "1.5B",
     "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
    # {"dataset": "rosbank", "config": "configs/rosbank.yaml", "model_size": "7B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    # {"dataset": "rosbank", "config": "configs/rosbank.yaml", "model_size": "14B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"},
]


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if "label_names" in config.get("dataset", {}):
        config["dataset"]["label_names"] = {
            str(k): v for k, v in config["dataset"]["label_names"].items()
        }
    return config


def run_one(run: dict) -> dict:
    """Загружает конфиг, подставляет модель и output_dir, запускает train()."""
    config = load_config(run["config"])
    config["lora"]["model_name"] = run["model_name"]
    # Отдельная папка для каждого запуска: results/gender_1.5B/, results/rosbank_7B/ и т.д.
    config["output"]["base_dir"] = f"results/{run['dataset']}_{run['model_size']}"
    config["output"]["metrics"]  = "metrics.json"
    Path(config["output"]["base_dir"]).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train(config)
    elapsed = time.time() - t0

    metrics_path = Path(config["output"]["base_dir"]) / config["output"]["metrics"]
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["elapsed_minutes"] = round(elapsed / 60, 1)
    return metrics


def main():
    total   = len(RUNS)
    results = {}

    for run in tqdm(RUNS, desc="LoRA runs", unit="model"):
        label = f"{run['dataset']}_{run['model_size']}"
        tqdm.write(f"\n{'='*55}")
        tqdm.write(f"  Запуск: {label}")
        tqdm.write(f"  Модель: {run['model_name']}")
        tqdm.write(f"  Конфиг: {run['config']}")
        tqdm.write(f"{'='*55}")

        try:
            metrics = run_one(run)
            results[label] = {"status": "ok", **metrics}
            tqdm.write(f"  Готово за {metrics.get('elapsed_minutes')} мин")
            tqdm.write(f"  Метрики: {metrics}")
        except Exception as e:
            results[label] = {"status": "error", "error": str(e)}
            tqdm.write(f"  ОШИБКА: {e}")

    print(f"\n{'='*55}")
    print("ИТОГ")
    print(f"{'='*55}")
    for label, res in results.items():
        status = res.get("status")
        time_  = res.get("elapsed_minutes", "?")
        metric_val = (
            res.get("eval_accuracy") or
            res.get("eval_roc_auc_ovr_macro") or "—"
        )
        print(f"  {label:<25} {status:<8} метрика={metric_val}  время={time_} мин")

    Path("results").mkdir(exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nСводка сохранена → results/summary.json")


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    main()
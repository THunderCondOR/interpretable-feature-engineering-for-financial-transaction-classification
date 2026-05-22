"""
run_all_loras.py

Последовательно обучает 4 LoRA модели:
    1. Gender  — 7B
    2. Gender  — 14B
    3. Age     — 7B
    4. Age     — 14B

Запуск:
    PYTHONPATH=. python run_all_loras.py
"""

import copy
import json
import time
import yaml
from pathlib import Path
from tqdm import tqdm

from src.models.lora_trainer import train


# ---------------------------------------------------------------------------
# Конфигурации четырёх запусков
# ---------------------------------------------------------------------------

RUNS = [
    # {"dataset": "gender", "config": "configs/gender.yaml", "model_size": "7B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    # {"dataset": "gender", "config": "configs/gender.yaml", "model_size": "14B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"},
    # {"dataset": "age",    "config": "configs/age.yaml",    "model_size": "7B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
    # {"dataset": "age",    "config": "configs/age.yaml",    "model_size": "14B",
    #  "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"},
     {"dataset": "gender", "config": "configs/gender.yaml", "model_size": "1.5B",
     "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
    {"dataset": "age",    "config": "configs/age.yaml",    "model_size": "1.5B",
     "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_one(run: dict) -> dict:
    """Загружает конфиг, подставляет модель и output_dir, запускает train()."""
    config = load_config(run["config"])

    # Подставляем модель из RUNS
    config["lora"]["model_name"] = run["model_name"]

    # Отдельная папка для каждого запуска: results/gender_7B/, results/age_14B/ и т.д.
    config["output"]["base_dir"] = f"results/{run['dataset']}_{run['model_size']}"
    config["output"]["metrics"]  = "metrics.json"

    Path(config["output"]["base_dir"]).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train(config)
    elapsed = time.time() - t0

    # Читаем метрики которые сохранил train()
    metrics_path = Path(config["output"]["base_dir"]) / config["output"]["metrics"]
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["elapsed_minutes"] = round(elapsed / 60, 1)

    return metrics


def main():
    total = len(RUNS)
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
            # Продолжаем следующий запуск

    # Итоговая таблица
    print(f"\n{'='*55}")
    print("ИТОГ")
    print(f"{'='*55}")
    for label, res in results.items():
        status = res.get("status")
        time_  = res.get("elapsed_minutes", "?")
        # Выводим основную метрику
        metric_val = (
            res.get("eval_accuracy") or
            res.get("eval_roc_auc_ovr_macro") or
            "—"
        )
        print(f"  {label:<20} {status:<8} метрика={metric_val}  время={time_} мин")

    # Сохраняем сводку
    with open("results/summary.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nСводка сохранена → results/summary.json")


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    main()

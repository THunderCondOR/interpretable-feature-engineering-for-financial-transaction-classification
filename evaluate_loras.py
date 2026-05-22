"""
evaluate_loras.py

Загружает сохранённые LoRA чекпоинты и считает набор метрик на test сплите.

Метрики:
    accuracy, f1_macro, f1_weighted, roc_auc_ovr_macro,
    cohen_kappa, matthews_corrcoef, per-class precision/recall/f1

Запуск:
    PYTHONPATH=. python evaluate_loras.py
    PYTHONPATH=. python evaluate_loras.py --splits val test
    PYTHONPATH=. python evaluate_loras.py --runs gender_7B age_14B
"""

import argparse
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.data.loader import load_dataset, add_features
from src.data.aggregator import build_user_summary_str


# ---------------------------------------------------------------------------
# Конфигурации запусков — должны совпадать с run_all_loras.py
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
    {"label": "gender_1.5B", "dataset": "gender", "config": "configs/gender.yaml", "model_size": "1.5B",
     "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
    {"label": "age_1.5B","dataset": "age",    "config": "configs/age.yaml",    "model_size": "1.5B",
     "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
]


# ---------------------------------------------------------------------------
# Загрузка модели из чекпоинта
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(base_model_name: str, adapter_dir: str, num_labels: int):
    print(f"  Загружаем базовую модель {base_model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        device_map="auto",
        quantization_config=bnb_config,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"  Накатываем LoRA адаптеры из {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def predict(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 16):
    """Возвращает (predicted_labels, probabilities)."""
    all_logits = []

    for i in tqdm(range(0, len(texts), batch_size), desc="    inference", leave=False):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            logits = model(**enc).logits
        all_logits.append(logits.cpu().float())

    logits    = torch.cat(all_logits, dim=0)
    probs     = torch.softmax(logits, dim=-1).numpy()
    preds     = logits.argmax(dim=-1).numpy()
    return preds, probs


# ---------------------------------------------------------------------------
# Сборка входных текстов
# ---------------------------------------------------------------------------

def build_texts(df: pd.DataFrame, config: dict, system_prompt: str) -> tuple[list, list]:
    """Возвращает (texts, labels) — по одному на клиента."""
    category_label = config["dataset"].get("category_label", "категории трат")
    label_names    = config["dataset"]["label_names"]
    options = ", ".join(f"{k} ({v})" for k, v in label_names.items())

    texts, labels = [], []
    for cid in df["customer_id"].unique():
        client_df = df[df["customer_id"] == cid]
        summary   = build_user_summary_str(client_df, category_label)
        text      = f"{system_prompt}\n\nДанные клиента:\n{summary}\n\nВарианты ответа: {options}."
        texts.append(text)
        labels.append(int(client_df["label"].iloc[0]))

    return texts, labels


# ---------------------------------------------------------------------------
# Подсчёт метрик
# ---------------------------------------------------------------------------

def compute_metrics(labels: list, preds: np.ndarray, probs: np.ndarray,
                    label_names: dict, num_labels: int) -> dict:
    labels = np.array(labels)
    names  = [label_names.get(str(i), label_names.get(i, str(i))) for i in range(num_labels)]

    metrics = {
        "accuracy":         round(accuracy_score(labels, preds), 4),
        "f1_macro":         round(f1_score(labels, preds, average="macro",    zero_division=0), 4),
        "f1_weighted":      round(f1_score(labels, preds, average="weighted", zero_division=0), 4),
        "cohen_kappa":      round(cohen_kappa_score(labels, preds), 4),
        "matthews_corrcoef":round(matthews_corrcoef(labels, preds), 4),
    }

    # ROC-AUC (только если probs есть для всех классов)
    try:
        if num_labels == 2:
            metrics["roc_auc"] = round(
                roc_auc_score(labels, probs[:, 1]), 4
            )
        else:
            metrics["roc_auc_ovr_macro"] = round(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro"), 4
            )
    except ValueError as e:
        metrics["roc_auc_error"] = str(e)

    # Per-class F1 / precision / recall
    report = classification_report(labels, preds, target_names=names,
                                   output_dict=True, zero_division=0)
    for cls_name in names:
        safe = cls_name.replace(" ", "_")
        metrics[f"precision_{safe}"] = round(report[cls_name]["precision"], 4)
        metrics[f"recall_{safe}"]    = round(report[cls_name]["recall"],    4)
        metrics[f"f1_{safe}"]        = round(report[cls_name]["f1-score"],  4)

    # Confusion matrix (как список для JSON)
    metrics["confusion_matrix"] = confusion_matrix(labels, preds).tolist()

    return metrics


# ---------------------------------------------------------------------------
# Один запуск
# ---------------------------------------------------------------------------

def evaluate_run(run: dict, splits: list[str]) -> dict:
    label      = run["label"]
    adapter_dir = Path("results") / label / "lora" / "final"

    if not adapter_dir.exists():
        return {"error": f"Чекпоинт не найден: {adapter_dir}"}

    with open(run["config"]) as f:
        config = yaml.safe_load(f)

    config["lora"]["model_name"] = run["model_name"]
    num_labels  = config["dataset"]["num_labels"]
    label_names = config["dataset"]["label_names"]
    max_length  = config["lora"].get("max_length", 1024)

    # System prompt
    sys_path      = Path(config["prompts"]["base_dir"]) / config["prompts"]["system"]
    system_prompt = sys_path.read_text(encoding="utf-8") if sys_path.exists() else ""

    # Загружаем модель один раз для всех сплитов
    model, tokenizer = load_model_and_tokenizer(
        run["model_name"], str(adapter_dir), num_labels
    )

    results = {}
    for split in splits:
        print(f"  Сплит: {split}")
        df     = add_features(load_dataset(config, split))
        texts, labels = build_texts(df, config, system_prompt)
        preds, probs  = predict(model, tokenizer, texts, max_length)
        metrics = compute_metrics(labels, preds, probs, label_names, num_labels)
        results[split] = metrics
        print(f"    accuracy={metrics['accuracy']}  f1_macro={metrics['f1_macro']}")

    # Освобождаем память
    del model
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", nargs="+",
        default=[f'{r["dataset"]}_{r["model_size"]}' for r in RUNS],
        help="Какие запуски оценивать (например: gender_7B age_14B)"
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=["val", "test"],
        help="Какие сплиты использовать"
    )
    args = parser.parse_args()

    runs_to_eval = [r for r in RUNS if f'{r["dataset"]}_{r["model_size"]}' in args.runs]
    all_results  = {}

    for run in tqdm(runs_to_eval, desc="Evaluating", unit="model"):
        print(f"\n{'='*55}")
        print(f"  {f'{run["dataset"]}_{run["model_size"]}'}")
        print(f"{'='*55}")

        result = evaluate_run(run, args.splits)
        all_results[run["label"]] = result

    # Сохраняем полные результаты
    out_path = Path("results/eval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nПолные метрики → {out_path}")

    # Печатаем сводную таблицу accuracy
    print(f"\n{'─'*65}")
    print(f"{'Модель':<20} {'Сплит':<8} {'Accuracy':<10} {'F1 macro':<10} {'ROC-AUC'}")
    print(f"{'─'*65}")
    for label, splits_res in all_results.items():
        if "error" in splits_res:
            print(f"{label:<20} ERROR: {splits_res['error']}")
            continue
        for split, m in splits_res.items():
            roc = m.get("roc_auc") or m.get("roc_auc_ovr_macro") or "—"
            print(f"{label:<20} {split:<8} {m.get('accuracy','—'):<10} "
                  f"{m.get('f1_macro','—'):<10} {roc}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    main()
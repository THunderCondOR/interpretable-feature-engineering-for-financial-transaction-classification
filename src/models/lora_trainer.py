"""
src/models/lora_trainer.py

LoRA fine-tuning для классификации транзакционного поведения.

Ключевые особенности:
  - Вход: user_summary_str (~500 токенов) — честное сравнение с few-shot LLM
  - num_labels, label-колонка и метрика читаются из конфига
  - Один файл работает для gender (accuracy), age (roc_auc_ovr_macro), rosbank (roc_auc_ovr_macro)
  - Диспетчер get_summary_fn() подбирает нужный агрегатор по имени датасета
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

import evaluate
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.data.loader import load_dataset, add_features
from src.data.aggregator import get_summary_fn


def _build_input_text(client_df: pd.DataFrame, config: dict, system_prompt: str) -> str:
    """
    Собирает входной текст для LoRA:
        [system_prompt]
        Данные клиента:
        [user_summary_str]
        Варианты ответа: 0 (label_0), 1 (label_1), ...
    """
    category_label = config["dataset"].get("category_label", "категории трат")
    summary_fn     = get_summary_fn(config)
    summary        = summary_fn(client_df, category_label)

    label_names = config["dataset"]["label_names"]
    options = ", ".join(
        f"{k} ({v})" for k, v in sorted(label_names.items(), key=lambda x: int(x[0]))
    )

    return f"{system_prompt}\n\nДанные клиента:\n{summary}\n\nВарианты ответа: {options}."


def _prepare_hf_dataset(
    df: pd.DataFrame,
    config: dict,
    system_prompt: str,
    tokenizer,
    max_length: int,
) -> Dataset:
    records = []
    for cid in df["customer_id"].unique():
        client_df = df[df["customer_id"] == cid]
        text  = _build_input_text(client_df, config, system_prompt)
        label = int(client_df["label"].iloc[0])
        records.append({"text": text, "label": label})

    hf_ds = Dataset.from_list(records)

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    return hf_ds.map(tokenize, batched=True, remove_columns=["text"])


def _make_compute_metrics(config: dict):
    """
    Возвращает функцию метрики в зависимости от конфига:
      accuracy          → accuracy
      roc_auc_ovr_macro → ROC-AUC (работает для бинарной и многоклассовой)
    """
    metric_name = config["dataset"]["metric"]
    num_labels  = config["dataset"]["num_labels"]

    if metric_name == "accuracy":
        acc = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            return acc.compute(predictions=preds, references=labels)
        return compute_metrics

    elif metric_name == "roc_auc_ovr_macro":
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            probs = torch.softmax(
                torch.tensor(logits, dtype=torch.float32), dim=-1
            ).numpy()
            try:
                if num_labels == 2:
                    score = roc_auc_score(labels, probs[:, 1])
                else:
                    score = roc_auc_score(
                        labels, probs, multi_class="ovr", average="macro"
                    )
            except ValueError:
                score = 0.0
            return {"roc_auc_ovr_macro": float(score)}
        return compute_metrics

    else:
        raise ValueError(
            f"Неизвестная метрика '{metric_name}'. "
            f"Поддерживаются: accuracy, roc_auc_ovr_macro"
        )


def train(config: dict) -> None:
    """
    Полный цикл обучения LoRA.
    """
    lora_cfg = config["lora"]
    out_dir  = Path(config["output"]["base_dir"]) / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Загружаем данные...")
    train_df = add_features(load_dataset(config, "train"))
    val_df   = add_features(load_dataset(config, "val"))
    print(f"  train: {train_df['customer_id'].nunique()} клиентов")
    print(f"  val:   {val_df['customer_id'].nunique()} клиентов")

    sys_path = Path(config["prompts"]["base_dir"]) / config["prompts"]["system"]
    system_prompt = sys_path.read_text(encoding="utf-8") if sys_path.exists() else ""

    print(f"Загружаем токенизатор: {lora_cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(lora_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_length = lora_cfg.get("max_length", 1024)

    print("Строим user_summary_str для каждого клиента...")
    train_dataset = _prepare_hf_dataset(train_df, config, system_prompt, tokenizer, max_length)
    val_dataset   = _prepare_hf_dataset(val_df,   config, system_prompt, tokenizer, max_length)
    print(f"  train: {len(train_dataset)} примеров")
    print(f"  val:   {len(val_dataset)} примеров")

    print("Загружаем модель (4-bit квантизация)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=lora_cfg.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        lora_cfg["model_name"],
        num_labels=config["dataset"]["num_labels"],
        device_map="auto",
        quantization_config=bnb_config,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg.get("lora_rank", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 8),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    grad_acc = lora_cfg.get("gradient_accumulation_steps", 8)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        learning_rate=lora_cfg.get("learning_rate", 2e-4),
        per_device_train_batch_size=lora_cfg.get("batch_size", 4),
        per_device_eval_batch_size=lora_cfg.get("batch_size", 4),
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=lora_cfg.get("num_epochs", 5),
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        fp16=True,
        max_grad_norm=1.0,
        optim="paged_adamw_32bit",
        report_to=[],
    )

    import transformers
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_make_compute_metrics(config),
    )
    version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if version >= (4, 46):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("Обучаем...")
    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Модель сохранена → {final_dir}")

    metrics = trainer.evaluate()
    metrics_path = Path(config["output"]["base_dir"]) / config["output"]["metrics"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Метрики: {metrics}")
    print(f"Сохранены → {metrics_path}")
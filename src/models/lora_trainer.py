"""
src/models/lora_trainer.py

LoRA fine-tuning for transaction-behavior classification.

This version always reports accuracy, even when another metric is also useful.
For Rosbank churn the primary metric in configs/rosbank.yaml is accuracy, but the
saved metrics also include balanced accuracy, macro/weighted F1, MCC, ROC-AUC
when probabilities are available, and the confusion matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.data.aggregator import get_summary_fn
from src.data.loader import add_features, load_dataset


def _build_input_text(client_df: pd.DataFrame, config: dict, system_prompt: str) -> str:
    """
    Build the input text for LoRA from one client's aggregated transaction profile.
    """
    category_label = config["dataset"].get("category_label", "категории трат")
    summary_fn = get_summary_fn(config)
    summary = summary_fn(client_df, category_label)

    label_names = {str(k): v for k, v in config["dataset"]["label_names"].items()}
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
    """One HF dataset row per client."""
    records = []
    for cid in df["customer_id"].unique():
        client_df = df[df["customer_id"] == cid]
        records.append({
            "text": _build_input_text(client_df, config, system_prompt),
            "label": int(client_df["label"].iloc[0]),
        })

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


def _classification_metrics(labels: np.ndarray, logits: np.ndarray, num_labels: int) -> dict:
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits)
    preds = logits.argmax(axis=-1)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }

    # ROC-AUC is auxiliary. It can fail if a split has one class only.
    try:
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
        if num_labels == 2:
            metrics["roc_auc"] = float(roc_auc_score(labels, probs[:, 1]))
        else:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
    except Exception as exc:
        metrics["roc_auc_error"] = str(exc)

    return metrics


def _make_compute_metrics(config: dict):
    """Trainer callback: always returns accuracy plus auxiliary metrics."""
    num_labels = int(config["dataset"]["num_labels"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return _classification_metrics(labels, logits, num_labels)

    return compute_metrics


def _load_system_prompt(config: dict) -> str:
    sys_path = Path(config["prompts"]["base_dir"]) / config["prompts"]["system"]
    if not sys_path.exists():
        raise FileNotFoundError(
            f"System prompt not found: {sys_path}. Check configs/*yaml prompts.base_dir/system."
        )
    return sys_path.read_text(encoding="utf-8")


def train(config: dict) -> None:
    """Train LoRA and evaluate on validation and test splits."""
    lora_cfg = config["lora"]
    out_dir = Path(config["output"]["base_dir"]) / "lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Загружаем данные...")
    train_df = add_features(load_dataset(config, "train"))
    val_df = add_features(load_dataset(config, "val"))
    test_df = add_features(load_dataset(config, "test"))
    print(f"  train: {train_df['customer_id'].nunique()} клиентов")
    print(f"  val:   {val_df['customer_id'].nunique()} клиентов")
    print(f"  test:  {test_df['customer_id'].nunique()} клиентов")

    system_prompt = _load_system_prompt(config)

    print(f"Загружаем токенизатор: {lora_cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(lora_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_length = int(lora_cfg.get("max_length", 1024))

    print("Строим user_summary_str для каждого клиента...")
    train_dataset = _prepare_hf_dataset(train_df, config, system_prompt, tokenizer, max_length)
    val_dataset = _prepare_hf_dataset(val_df, config, system_prompt, tokenizer, max_length)
    test_dataset = _prepare_hf_dataset(test_df, config, system_prompt, tokenizer, max_length)
    print(f"  train: {len(train_dataset)} примеров")
    print(f"  val:   {len(val_dataset)} примеров")
    print(f"  test:  {len(test_dataset)} примеров")

    print("Загружаем модель (4-bit квантизация)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=lora_cfg.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        lora_cfg["model_name"],
        num_labels=int(config["dataset"]["num_labels"]),
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

    metric_for_best_model = config["dataset"].get("metric", "accuracy")
    if metric_for_best_model not in {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc"}:
        metric_for_best_model = "accuracy"

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        learning_rate=lora_cfg.get("learning_rate", 2e-4),
        per_device_train_batch_size=lora_cfg.get("batch_size", 4),
        per_device_eval_batch_size=lora_cfg.get("batch_size", 4),
        gradient_accumulation_steps=lora_cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=lora_cfg.get("num_epochs", 5),
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        fp16=True,
        max_grad_norm=1.0,
        optim="paged_adamw_32bit",
        report_to=[],
        load_best_model_at_end=False,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
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

    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    metrics = {**val_metrics, **test_metrics}

    # Convenience aliases for quick reading in summary scripts.
    if "val_accuracy" in metrics:
        metrics["accuracy_val"] = metrics["val_accuracy"]
    if "test_accuracy" in metrics:
        metrics["accuracy_test"] = metrics["test_accuracy"]

    metrics_path = Path(config["output"]["base_dir"]) / config["output"]["metrics"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Метрики: {metrics}")
    print(f"Сохранены → {metrics_path}")

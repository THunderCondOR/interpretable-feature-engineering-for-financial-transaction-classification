"""
src/models/lora_trainer.py

Sequential LoRA fine-tuning for transaction-behavior classification.
A dataset config may define several LoRA runs under lora.models; they are trained
one by one without editing the config between runs.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

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


RESERVED_LORA_KEYS = {"defaults", "models"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "model"


def _resolve_lora_runs(config: dict, model_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Resolve lora.defaults + lora.models into concrete run configs."""
    lora_cfg = config["lora"]
    defaults = {k: v for k, v in lora_cfg.items() if k not in RESERVED_LORA_KEYS}
    defaults.update(lora_cfg.get("defaults", {}))

    raw_models = lora_cfg.get("models")
    if raw_models is None:
        if "model_name" not in defaults:
            raise ValueError("Config must define either lora.model_name or lora.models.")
        raw_models = [{"name": defaults.get("name"), "model_name": defaults["model_name"]}]

    runs = []
    for item in raw_models:
        run = {**defaults, **item}
        if "model_name" not in run:
            raise ValueError(f"LoRA run is missing model_name: {item}")
        run["name"] = run.get("name") or _slugify(run["model_name"])
        runs.append(run)

    if model_names is None:
        return runs

    requested = set(model_names)
    selected = [run for run in runs if run["name"] in requested or run["model_name"] in requested]
    found = {run["name"] for run in selected} | {run["model_name"] for run in selected}
    unknown = requested - found
    if unknown:
        available = ", ".join(run["name"] for run in runs)
        raise ValueError(f"Unknown LoRA model(s): {sorted(unknown)}. Available names: {available}")
    return selected


def _build_input_text(client_df: pd.DataFrame, config: dict, system_prompt: str) -> str:
    """Build the input text for LoRA from one client's aggregated transaction profile."""
    category_label = config["dataset"].get(
        "category_label", "transaction categories"
    )
    summary_fn = get_summary_fn(config)
    summary = summary_fn(client_df, category_label)

    label_names = {str(k): v for k, v in config["dataset"]["label_names"].items()}
    options = ", ".join(
        f"{k} ({v})" for k, v in sorted(label_names.items(), key=lambda x: int(x[0]))
    )

    return (
        f"{system_prompt}\n\nCLIENT TRANSACTION PROFILE\n{summary}\n\n"
        f"ALLOWED LABELS: {options}."
    )


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


def _build_training_args(run_dir: Path, run_cfg: dict, metric_for_best_model: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy="steps",
        eval_steps=int(run_cfg.get("eval_steps", 50)),
        save_strategy="epoch",
        learning_rate=float(run_cfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(run_cfg.get("batch_size", 4)),
        per_device_eval_batch_size=int(run_cfg.get("eval_batch_size", run_cfg.get("batch_size", 4))),
        gradient_accumulation_steps=int(run_cfg.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(run_cfg.get("num_epochs", 5)),
        weight_decay=float(run_cfg.get("weight_decay", 0.01)),
        warmup_steps=int(run_cfg.get("warmup_steps", 50)),
        logging_steps=int(run_cfg.get("logging_steps", 10)),
        fp16=bool(run_cfg.get("fp16", True)),
        bf16=bool(run_cfg.get("bf16", False)),
        max_grad_norm=float(run_cfg.get("max_grad_norm", 1.0)),
        optim=str(run_cfg.get("optim", "paged_adamw_32bit")),
        report_to=[],
        load_best_model_at_end=False,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
    )


def _train_one(
    config: dict,
    run_cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    system_prompt: str,
) -> dict:
    run_name = run_cfg["name"]
    model_name = run_cfg["model_name"]
    run_dir = Path(config["output"]["base_dir"]) / "lora" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}\nLoRA run: {run_name}\nmodel: {model_name}\n{'=' * 72}")
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_length = int(run_cfg.get("max_length", 1024))

    print("Building one training example per client...")
    train_dataset = _prepare_hf_dataset(train_df, config, system_prompt, tokenizer, max_length)
    val_dataset = _prepare_hf_dataset(val_df, config, system_prompt, tokenizer, max_length)
    test_dataset = _prepare_hf_dataset(test_df, config, system_prompt, tokenizer, max_length)
    print(f"  train: {len(train_dataset)} examples")
    print(f"  val:   {len(val_dataset)} examples")
    print(f"  test:  {len(test_dataset)} examples")

    load_in_4bit = bool(run_cfg.get("load_in_4bit", True))
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=int(config["dataset"]["num_labels"]),
        device_map="auto",
        quantization_config=quantization_config,
        pad_token_id=tokenizer.pad_token_id,
    )
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(run_cfg.get("lora_rank", 8)),
        lora_alpha=int(run_cfg.get("lora_alpha", 8)),
        target_modules=run_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=float(run_cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    metric_for_best_model = config["dataset"].get("metric", "accuracy")
    if metric_for_best_model not in {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc"}:
        metric_for_best_model = "accuracy"

    training_args = _build_training_args(run_dir, run_cfg, metric_for_best_model)

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

    print("Training...")
    trainer.train()

    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved model -> {final_dir}")

    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    metrics = {
        "run_name": run_name,
        "model_name": model_name,
        "config": {k: v for k, v in run_cfg.items() if k != "models"},
        **val_metrics,
        **test_metrics,
    }

    if "val_accuracy" in metrics:
        metrics["accuracy_val"] = metrics["val_accuracy"]
    if "test_accuracy" in metrics:
        metrics["accuracy_test"] = metrics["test_accuracy"]

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics -> {metrics_path}")

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def train(config: dict, model_names: list[str] | None = None) -> None:
    """Train one or more LoRA runs and evaluate each on validation and test splits."""
    runs = _resolve_lora_runs(config, model_names=model_names)
    print("Loading data...")
    train_df = add_features(load_dataset(config, "train"))
    val_df = add_features(load_dataset(config, "val"))
    test_df = add_features(load_dataset(config, "test"))
    print(f"  train: {train_df['customer_id'].nunique()} clients")
    print(f"  val:   {val_df['customer_id'].nunique()} clients")
    print(f"  test:  {test_df['customer_id'].nunique()} clients")

    system_prompt = _load_system_prompt(config)
    summary = {}
    summary_path = Path(config["output"]["base_dir"]) / "lora_metrics.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

    for run_cfg in runs:
        metrics = _train_one(config, run_cfg, train_df, val_df, test_df, system_prompt)
        summary[run_cfg["name"]] = metrics
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Updated LoRA summary -> {summary_path}")

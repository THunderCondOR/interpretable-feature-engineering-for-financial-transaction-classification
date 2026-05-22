"""
prepare_age.py

Скачивает age датасет с HuggingFace (pytorch-lifestream/age-group-prediction),
мержит транзакции с лейблами (bins → age_group, 0-индекс),
делает сплит 70/10/20 по клиентам,
сохраняет train.csv / val.csv / test.csv в data/age/.

Датасет большой (~26M строк), скачивание занимает 5-10 минут.

Запуск:
    python prepare_age.py
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

Path("data/age").mkdir(parents=True, exist_ok=True)

# ── Транзакции ───────────────────────────────────────────────────────────────
print("Загружаем транзакции (pytorch-lifestream/age-group-prediction)...")
print("Это займёт несколько минут (~26M строк)...")
trx = load_dataset(
    "pytorch-lifestream/age-group-prediction",
    "transactions_train",
    split="train",
).to_pandas()
print(f"  Строк: {len(trx):,}, колонки: {list(trx.columns)}")

# ── Лейблы ───────────────────────────────────────────────────────────────────
print("Загружаем лейблы (train_target)...")
labels = load_dataset(
    "pytorch-lifestream/age-group-prediction",
    "train_target",
    split="train",
).to_pandas()
print(f"  Клиентов: {len(labels)}, колонки: {list(labels.columns)}")

# Определяем колонки динамически — на случай если названия отличаются
id_col    = next(c for c in labels.columns if "id" in c.lower())
label_col = next(c for c in labels.columns if c != id_col)
print(f"  ID колонка: '{id_col}', label колонка: '{label_col}'")
print(f"  Распределение {label_col}:\n{labels[label_col].value_counts().sort_index().to_string()}")

# ── Merge ────────────────────────────────────────────────────────────────────
df = trx.merge(labels[[id_col, label_col]], on=id_col, how="inner")
print(f"\nПосле merge: {len(df):,} строк, {df[id_col].nunique()} клиентов")

# bins идут с 1 (1,2,3,4) → приводим к 0-индексу (0,1,2,3)
if df[label_col].min() == 1:
    df[label_col] = df[label_col] - 1
    print(f"bins сдвинуты: {sorted(df[label_col].unique())} (0-indexed)")

# Унифицируем имена для конфига
df = df.rename(columns={
    id_col:    "customer_id",
    label_col: "age_group",
})
print(f"Итоговые колонки: {list(df.columns)}")

# ── Сплит 70 / 10 / 20 по клиентам ──────────────────────────────────────────
clients = df["customer_id"].unique()
train_ids, temp     = train_test_split(clients, test_size=0.30,  random_state=42)
val_ids,   test_ids = train_test_split(temp,    test_size=0.667, random_state=42)

train_df = df[df["customer_id"].isin(train_ids)]
val_df   = df[df["customer_id"].isin(val_ids)]
test_df  = df[df["customer_id"].isin(test_ids)]

print("Сохраняем CSV (может занять минуту из-за размера)...")
train_df.to_csv("data/age/train.csv", index=False)
val_df.to_csv(  "data/age/val.csv",   index=False)
test_df.to_csv( "data/age/test.csv",  index=False)

print(f"\nСохранено в data/age/:")
print(f"  train.csv: {train_df['customer_id'].nunique()} клиентов, {len(train_df):,} строк")
print(f"  val.csv:   {val_df['customer_id'].nunique()} клиентов, {len(val_df):,} строк")
print(f"  test.csv:  {test_df['customer_id'].nunique()} клиентов, {len(test_df):,} строк")
print(f"\n>>> Скопируй итоговые колонки в configs/age.yaml секцию dataset.columns")
print(f"    Колонки: {list(df.columns)}")

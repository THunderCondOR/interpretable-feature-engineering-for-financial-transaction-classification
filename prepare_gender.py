"""
prepare_gender.py

Скачивает gender датасет с HuggingFace (pytorch-lifestream/transactions-gender),
мержит транзакции с лейблами, делает сплит 70/10/20 по клиентам,
сохраняет train.csv / val.csv / test.csv в data/gender/.

Запуск:
    python prepare_gender.py
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

Path("data/gender").mkdir(parents=True, exist_ok=True)

# ── Транзакции ───────────────────────────────────────────────────────────────
print("Загружаем транзакции (pytorch-lifestream/transactions-gender)...")
trx = load_dataset(
    "pytorch-lifestream/transactions-gender",
    "transactions_data",
    split="train",
).to_pandas()
print(f"  Строк: {len(trx):,}, колонки: {list(trx.columns)}")

# ── Лейблы ───────────────────────────────────────────────────────────────────
print("Загружаем лейблы...")
labels = load_dataset(
    "pytorch-lifestream/transactions-gender",
    "labels",
    split="train",
).to_pandas()
print(f"  Клиентов: {len(labels)}, колонки: {list(labels.columns)}")
print(f"  Распределение gender:\n{labels['gender'].value_counts().to_string()}")

# ── Merge ────────────────────────────────────────────────────────────────────
df = trx.merge(labels, on="customer_id", how="inner")
print(f"\nПосле merge: {len(df):,} строк, {df['customer_id'].nunique()} клиентов")
print(f"Итоговые колонки: {list(df.columns)}")

# ── Сплит 70 / 10 / 20 по клиентам ──────────────────────────────────────────
clients = df["customer_id"].unique()
train_ids, temp    = train_test_split(clients, test_size=0.30,  random_state=42)
val_ids,   test_ids = train_test_split(temp,   test_size=0.667, random_state=42)

train_df = df[df["customer_id"].isin(train_ids)]
val_df   = df[df["customer_id"].isin(val_ids)]
test_df  = df[df["customer_id"].isin(test_ids)]

train_df.to_csv("data/gender/train.csv", index=False)
val_df.to_csv(  "data/gender/val.csv",   index=False)
test_df.to_csv( "data/gender/test.csv",  index=False)

print(f"\nСохранено в data/gender/:")
print(f"  train.csv: {train_df['customer_id'].nunique()} клиентов, {len(train_df):,} строк")
print(f"  val.csv:   {val_df['customer_id'].nunique()} клиентов, {len(val_df):,} строк")
print(f"  test.csv:  {test_df['customer_id'].nunique()} клиентов, {len(test_df):,} строк")
print(f"\n>>> Скопируй итоговые колонки в configs/gender.yaml секцию dataset.columns")
print(f"    Колонки: {list(df.columns)}")

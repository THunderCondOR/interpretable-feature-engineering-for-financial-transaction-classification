"""
prepare_age.py

Downloads the age-group dataset from HuggingFace
(pytorch-lifestream/age-group-prediction), merges transactions with labels,
converts labels to zero-based indices, creates a 70/10/20 client-level split,
and writes train/val/test CSV files.

The dataset is large, so downloading and conversion can take several minutes.

Run:
    python prepare_age.py
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

Path("data/age").mkdir(parents=True, exist_ok=True)

print("Loading transactions (pytorch-lifestream/age-group-prediction)...")
print("This may take several minutes for the full transaction table...")
trx = load_dataset(
    "pytorch-lifestream/age-group-prediction",
    "transactions_train",
    split="train",
).to_pandas()
print(f"  rows: {len(trx):,}, columns: {list(trx.columns)}")

print("Loading labels (train_target)...")
labels = load_dataset(
    "pytorch-lifestream/age-group-prediction",
    "train_target",
    split="train",
).to_pandas()
print(f"  clients: {len(labels)}, columns: {list(labels.columns)}")

id_col = next(c for c in labels.columns if "id" in c.lower())
label_col = next(c for c in labels.columns if c != id_col)
print(f"  id column: '{id_col}', label column: '{label_col}'")
print(f"  {label_col} distribution:\n{labels[label_col].value_counts().sort_index().to_string()}")

df = trx.merge(labels[[id_col, label_col]], on=id_col, how="inner")
print(f"\nAfter merge: {len(df):,} rows, {df[id_col].nunique()} clients")

if df[label_col].min() == 1:
    df[label_col] = df[label_col] - 1
    print(f"Labels shifted to zero-based indices: {sorted(df[label_col].unique())}")

df = df.rename(columns={
    id_col: "customer_id",
    label_col: "age_group",
})
print(f"Final columns: {list(df.columns)}")

clients = df["customer_id"].unique()
train_ids, temp = train_test_split(clients, test_size=0.30, random_state=42)
val_ids, test_ids = train_test_split(temp, test_size=0.667, random_state=42)

train_df = df[df["customer_id"].isin(train_ids)]
val_df = df[df["customer_id"].isin(val_ids)]
test_df = df[df["customer_id"].isin(test_ids)]

print("Writing CSV files...")
train_df.to_csv("data/age/train.csv", index=False)
val_df.to_csv("data/age/val.csv", index=False)
test_df.to_csv("data/age/test.csv", index=False)

print("\nSaved to data/age/:")
print(f"  train.csv: {train_df['customer_id'].nunique()} clients, {len(train_df):,} rows")
print(f"  val.csv:   {val_df['customer_id'].nunique()} clients, {len(val_df):,} rows")
print(f"  test.csv:  {test_df['customer_id'].nunique()} clients, {len(test_df):,} rows")
print("\nCopy the final columns to configs/age.yaml if the schema changes.")
print(f"Columns: {list(df.columns)}")

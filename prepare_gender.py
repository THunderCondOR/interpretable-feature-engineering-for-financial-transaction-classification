"""
prepare_gender.py

Downloads the gender dataset from HuggingFace
(pytorch-lifestream/transactions-gender), merges transactions with labels,
creates a 70/10/20 client-level split, and writes train/val/test CSV files.

Run:
    python prepare_gender.py
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

Path("data/gender").mkdir(parents=True, exist_ok=True)

print("Loading transactions (pytorch-lifestream/transactions-gender)...")
trx = load_dataset(
    "pytorch-lifestream/transactions-gender",
    "transactions_data",
    split="train",
).to_pandas()
print(f"  rows: {len(trx):,}, columns: {list(trx.columns)}")

print("Loading labels...")
labels = load_dataset(
    "pytorch-lifestream/transactions-gender",
    "labels",
    split="train",
).to_pandas()
print(f"  clients: {len(labels)}, columns: {list(labels.columns)}")
print(f"  gender distribution:\n{labels['gender'].value_counts().to_string()}")

df = trx.merge(labels, on="customer_id", how="inner")
print(f"\nAfter merge: {len(df):,} rows, {df['customer_id'].nunique()} clients")
print(f"Final columns: {list(df.columns)}")

clients = df["customer_id"].unique()
train_ids, temp = train_test_split(clients, test_size=0.30, random_state=42)
val_ids, test_ids = train_test_split(temp, test_size=0.667, random_state=42)

train_df = df[df["customer_id"].isin(train_ids)]
val_df = df[df["customer_id"].isin(val_ids)]
test_df = df[df["customer_id"].isin(test_ids)]

train_df.to_csv("data/gender/train.csv", index=False)
val_df.to_csv("data/gender/val.csv", index=False)
test_df.to_csv("data/gender/test.csv", index=False)

print("\nSaved to data/gender/:")
print(f"  train.csv: {train_df['customer_id'].nunique()} clients, {len(train_df):,} rows")
print(f"  val.csv:   {val_df['customer_id'].nunique()} clients, {len(val_df):,} rows")
print(f"  test.csv:  {test_df['customer_id'].nunique()} clients, {len(test_df):,} rows")
print("\nCopy the final columns to configs/gender.yaml if the schema changes.")
print(f"Columns: {list(df.columns)}")

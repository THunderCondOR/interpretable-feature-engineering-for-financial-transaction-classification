from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_ids(path: Path, column: str) -> set:
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in {path}")
    values = df[column].dropna()
    try:
        values = values.astype("int64")
    except Exception:
        values = values.astype(str)
    return set(values.tolist())


def split_ids(path: Path) -> set:
    df = pd.read_csv(path, usecols=["customer_id"])
    values = df["customer_id"].dropna()
    try:
        values = values.astype("int64")
    except Exception:
        values = values.astype(str)
    return set(values.tolist())


def client_label_counts(path: Path) -> dict:
    df = pd.read_csv(path, usecols=["customer_id", "label"])
    labels = df.drop_duplicates("customer_id")["label"]
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}


def check_dataset(dataset: str, data_root: Path, test_ids_dir: Path) -> None:
    id_columns = {"gender": "customer_id", "age": "client_id", "rosbank": "cl_id"}
    base = data_root / dataset
    train_path = base / "train.csv"
    val_path = base / "val.csv"
    test_path = base / "test.csv"
    test_ids_path = test_ids_dir / f"{dataset}_test_ids.csv"

    for path in [train_path, val_path, test_path, test_ids_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    train_ids = split_ids(train_path)
    val_ids = split_ids(val_path)
    test_ids = split_ids(test_path)
    official_ids = load_ids(test_ids_path, id_columns[dataset])

    overlap = {
        "train_val": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "val_test": len(val_ids & test_ids),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Client leakage for {dataset}: {overlap}")

    if official_ids != test_ids:
        raise RuntimeError(
            f"Official test ids do not match prepared test for {dataset}: "
            f"missing={len(official_ids - test_ids)}, extra={len(test_ids - official_ids)}"
        )

    print("\n" + "=" * 80)
    print(dataset)
    print("=" * 80)
    print(f"overlap: {overlap}")
    for name, path, ids in [("train", train_path, train_ids), ("val", val_path, val_ids), ("test", test_path, test_ids)]:
        counts = client_label_counts(path)
        majority = max(counts.values()) / sum(counts.values())
        rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        print(f"{name:<5}: rows={rows:,}, clients={len(ids):,}, labels={counts}, majority={majority:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "gender", "age", "rosbank"], default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--test-ids-dir", type=Path, default=Path("data/test_ids"))
    args = parser.parse_args()

    datasets = ["gender", "age", "rosbank"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        check_dataset(dataset, args.data_root, args.test_ids_dir)


if __name__ == "__main__":
    main()

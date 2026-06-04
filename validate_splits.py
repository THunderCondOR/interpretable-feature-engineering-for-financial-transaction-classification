from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TEST_ID_COLUMNS = {"gender": "customer_id", "age": "client_id", "rosbank": "cl_id"}


def ids_from_csv(path: Path, column: str) -> set[int]:
    return set(pd.read_csv(path, usecols=[column])[column].astype("int64"))


def prepared_ids(path: Path) -> set[int]:
    return ids_from_csv(path, "customer_id")


def label_counts(path: Path) -> dict[str, int]:
    df = pd.read_csv(path, usecols=["customer_id", "label"])
    labels = df.drop_duplicates("customer_id")["label"]
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}


def count_rows(path: Path) -> int:
    return sum(1 for _ in open(path, encoding="utf-8")) - 1


def check_dataset(dataset: str, data_dir: Path, test_ids_dir: Path) -> None:
    base = data_dir / dataset
    train_path = base / "train.csv"
    val_path = base / "val.csv"
    test_path = base / "test.csv"
    test_ids_path = test_ids_dir / f"{dataset}_test_ids.csv"

    train_ids = prepared_ids(train_path)
    val_ids = prepared_ids(val_path)
    test_ids = prepared_ids(test_path)
    expected_test_ids = ids_from_csv(test_ids_path, TEST_ID_COLUMNS[dataset])

    overlap = {
        "train_val": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "val_test": len(val_ids & test_ids),
    }
    assert overlap == {"train_val": 0, "train_test": 0, "val_test": 0}, overlap
    assert test_ids == expected_test_ids, (len(expected_test_ids - test_ids), len(test_ids - expected_test_ids))

    print("\n" + "=" * 80)
    print(dataset)
    print("=" * 80)
    print(f"overlap: {overlap}")
    for split_name, path, ids in [("train", train_path, train_ids), ("val", val_path, val_ids), ("test", test_path, test_ids)]:
        counts = label_counts(path)
        majority = max(counts.values()) / sum(counts.values())
        print(f"{split_name:<5}: rows={count_rows(path):,}, clients={len(ids):,}, labels={counts}, majority={majority:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared train/val/test splits.")
    parser.add_argument("--dataset", choices=["all", "gender", "age", "rosbank"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-ids-dir", type=Path, default=Path("data/test_ids"))
    args = parser.parse_args()

    datasets = ["gender", "age", "rosbank"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        check_dataset(dataset, args.data_dir, args.test_ids_dir)


if __name__ == "__main__":
    main()

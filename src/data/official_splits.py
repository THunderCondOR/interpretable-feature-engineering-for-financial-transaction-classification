"""
Utilities for preparing fixed client-level train/validation/test splits.

The downstream benchmark uses externally provided test client identifiers.
For each dataset, the official test clients are removed from the labeled pool,
and the remaining labeled clients are split into train/validation subsets.

The module writes normalized CSV files that follow the repository internal schema:
    customer_id, tr_datetime, amount, mcc_code_desc, label
Additional source-specific columns are preserved when they are useful for feature
engineering, for example Rosbank transaction category and currency names.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DEFAULT_VAL_SIZE_FROM_REMAINING = 1.0 / 9.0  # 10% val and 80% train if test is 10%.


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_name: str
    output_dir: Path
    test_ids_path: Path
    test_id_column: str
    label_column: str
    loader: Callable[[], tuple[pd.DataFrame, pd.DataFrame]]
    normalizer: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


CURRENCY_MAP = {
    810: "Рубль",
    643: "Рубль",
    978: "Евро",
    840: "Доллар США",
    826: "Фунт стерлингов",
    756: "Швейцарский франк",
    156: "Китайский юань",
    392: "Японская иена",
    980: "Украинская гривна",
    398: "Казахстанский тенге",
}

TRX_CATEGORY_MAP = {
    "POS": "оплата картой",
    "WD_ATM_PARTNER": "снятие наличных (банкомат партнера)",
    "WD_ATM_ROS": "снятие наличных (банкомат Росбанка)",
    "WD_ATM_OTHER": "снятие наличных (другой банк)",
    "CAT": "снятие наличных через кассу",
    "DEPOSIT": "пополнение счета",
    "C2C_OUT": "перевод на карту",
    "C2C_IN": "входящий перевод с карты",
    "MBO": "мобильный банк",
    "WEB": "интернет-банк",
    "BACK": "возврат средств",
}

MCC_TO_DESC = {
    4111: "Городской транспорт", 4112: "Железная дорога", 4121: "Такси", 4131: "Автобусы",
    4511: "Авиабилеты", 4722: "Туристические агентства", 4812: "Телеком", 4814: "Телеком",
    4816: "Интернет-сервисы", 4829: "Денежные переводы", 4899: "Кабельные и платные сервисы",
    4900: "ЖКХ", 5122: "Лекарства", 5200: "Строительные материалы", 5211: "Стройматериалы",
    5231: "Стекло и краски", 5251: "Скобяные товары", 5261: "Садовые принадлежности",
    5310: "Универсальные магазины", 5311: "Универмаги", 5331: "Дисконт-магазины",
    5399: "Разные магазины", 5411: "Супермаркеты", 5412: "Продуктовые магазины",
    5422: "Мясные магазины", 5441: "Кондитерские", 5451: "Молочные продукты", 5462: "Булочные",
    5499: "Продовольственные магазины", 5511: "Автосалоны", 5531: "Автозапчасти",
    5532: "Автошины", 5533: "Автотовары", 5541: "АЗС", 5542: "АЗС самообслуживания",
    5599: "Автоуслуги", 5611: "Мужская одежда", 5621: "Женская одежда", 5631: "Женские аксессуары",
    5641: "Детская одежда", 5651: "Одежда", 5661: "Обувь", 5691: "Одежда",
    5699: "Одежда и аксессуары", 5712: "Мебель", 5713: "Напольные покрытия", 5714: "Ткани",
    5719: "Товары для дома", 5722: "Бытовая техника", 5732: "Электроника", 5733: "Музыкальные товары",
    5734: "Компьютеры и ПО", 5735: "Музыка и медиа", 5811: "Кейтеринг", 5812: "Рестораны",
    5813: "Бары и клубы", 5814: "Фастфуд", 5912: "Аптеки", 5921: "Алкоголь",
    5931: "Секонд-хенд", 5932: "Антиквариат", 5941: "Спорттовары", 5942: "Книжные магазины",
    5943: "Канцелярские товары", 5944: "Ювелирные изделия", 5945: "Игрушки", 5946: "Фототовары",
    5947: "Подарки и сувениры", 5948: "Кожгалантерея", 5949: "Швейные товары", 5960: "Прямой маркетинг",
    5964: "Каталожные продажи", 5965: "Телемаркетинг", 5967: "Входящий телемаркетинг",
    5968: "Подписки", 5969: "Прямой маркетинг", 5970: "Товары для творчества", 5971: "Галереи",
    5977: "Косметика", 5992: "Цветы", 5993: "Табачные товары", 5994: "Газеты и журналы",
    5995: "Зоотовары", 5999: "Разные специализированные магазины", 6010: "Финансовые услуги кассы",
    6011: "Снятие наличных", 6012: "Финансовые услуги", 6051: "Денежные переводы",
    6211: "Брокеры и ценные бумаги", 7011: "Отели", 7210: "Прачечные", 7216: "Химчистки",
    7221: "Фотостудии", 7230: "Салоны красоты", 7273: "Знакомства", 7298: "Фитнес и здоровье",
    7299: "Персональные услуги", 7311: "Реклама", 7399: "Бизнес-услуги", 7512: "Аренда авто",
    7523: "Парковки", 7531: "Автосервис", 7534: "Шины", 7538: "Ремонт авто",
    7542: "Автомойки", 7629: "Ремонт техники", 7832: "Кинотеатры", 7841: "Видеопрокат",
    7911: "Танцевальные студии", 7922: "Театры и концерты", 7932: "Бильярд", 7933: "Боулинг",
    7991: "Туристические услуги", 7994: "Видеоигры", 7995: "Азартные игры", 7996: "Парки развлечений",
    7997: "Клубы", 7999: "Развлечения", 8011: "Врачи", 8021: "Стоматология",
    8043: "Оптика", 8062: "Больницы", 8099: "Медицина", 8211: "Школы", 8220: "Колледжи",
    8244: "Бизнес-образование", 8299: "Образование", 8999: "Профессиональные услуги", 9399: "Госуслуги",
}


def normalize_id_series(series: pd.Series) -> pd.Series:
    """Return ids in a stable comparable form."""
    values = series.dropna()
    try:
        return values.astype("int64")
    except Exception:
        return values.astype(str)


def read_id_set(path: Path, id_column: str) -> set[int | str]:
    ids_df = pd.read_csv(path)
    if id_column not in ids_df.columns:
        raise ValueError(f"Column {id_column!r} not found in {path}; columns={list(ids_df.columns)}")
    return set(normalize_id_series(ids_df[id_column]).tolist())


def parse_rosbank_datetime(value: Any) -> pd.Timestamp:
    """Convert strings like '21OCT17:13:08:28' to pandas Timestamp."""
    if pd.isna(value):
        return pd.NaT
    text = str(value)
    months = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
        "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    match = re.match(r"^(\d{2})([A-Z]{3})(\d{2}):(\d{2}:\d{2}:\d{2})$", text)
    if match:
        day, month, year, time = match.groups()
        return pd.to_datetime(f"20{year}-{months.get(month, '01')}-{day} {time}", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def mcc_to_text(value: Any) -> str:
    try:
        return MCC_TO_DESC.get(int(value), f"MCC {int(value)}")
    except Exception:
        return f"MCC {value}"


def load_gender_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = load_dataset(
        "pytorch-lifestream/transactions-gender",
        "transactions_data",
        split="train",
    ).to_pandas()
    labels = load_dataset(
        "pytorch-lifestream/transactions-gender",
        "labels",
        split="train",
    ).to_pandas()
    return transactions, labels


def normalize_gender(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = transactions.merge(labels[["customer_id", "gender"]], on="customer_id", how="inner")
    df = df.rename(columns={"gender": "label"})
    df["mcc_code_desc"] = df["mcc_code"].apply(mcc_to_text) if "mcc_code" in df.columns else "unknown category"
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["label"] = df["label"].astype(int)
    keep = ["customer_id", "tr_datetime", "amount", "mcc_code_desc", "label"]
    optional = [c for c in ["mcc_code", "tr_type", "term_id"] if c in df.columns]
    return df[keep + optional].copy()


def load_age_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = load_dataset(
        "pytorch-lifestream/age-group-prediction",
        "transactions_train",
        split="train",
    ).to_pandas()
    labels = load_dataset(
        "pytorch-lifestream/age-group-prediction",
        "train_target",
        split="train",
    ).to_pandas()
    return transactions, labels


def normalize_age(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = transactions.merge(labels[["client_id", "bins"]], on="client_id", how="inner")
    df = df.rename(
        columns={
            "client_id": "customer_id",
            "trans_date": "tr_datetime",
            "small_group": "mcc_code_desc",
            "amount_rur": "amount",
            "bins": "label",
        }
    )
    if pd.api.types.is_numeric_dtype(df["tr_datetime"]):
        df["tr_datetime"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["tr_datetime"].astype(int), unit="D")
    else:
        df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    df["mcc_code_desc"] = df["mcc_code_desc"].map(lambda x: f"operation group {x}")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["label"] = df["label"].astype(int)
    keep = ["customer_id", "tr_datetime", "amount", "mcc_code_desc", "label"]
    return df[keep].copy()


def load_rosbank_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        raw = load_dataset("pytorch-lifestream/rosbank-churn", "train", split="train").to_pandas()
    except Exception as first_error:
        try:
            raw = load_dataset("pytorch-lifestream/rosbank-churn", split="train").to_pandas()
        except Exception:
            raise first_error
    return raw, raw[["cl_id", "target_flag"]].drop_duplicates("cl_id").copy()


def normalize_rosbank(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    required = {"cl_id", "MCC", "currency", "TRDATETIME", "amount", "trx_category", "target_flag"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing required Rosbank columns: {sorted(missing)}")

    df = transactions.copy()
    df["tr_datetime"] = df["TRDATETIME"].apply(parse_rosbank_datetime)
    df["currency_name"] = df["currency"].map(
        lambda x: CURRENCY_MAP.get(int(x), f"Валюта {x}") if pd.notna(x) else "неизвестная валюта"
    )
    df["trx_cat_ru"] = df["trx_category"].map(TRX_CATEGORY_MAP).fillna(df["trx_category"].astype(str))
    df["mcc_desc"] = df["MCC"].apply(mcc_to_text)
    df["mcc_code_desc"] = df["mcc_desc"] + " [" + df["trx_cat_ru"] + "]"
    df = df.rename(columns={"cl_id": "customer_id", "target_flag": "label"})
    df["customer_id"] = df["customer_id"].astype(int)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["label"] = df["label"].astype(int)
    keep = [
        "customer_id", "tr_datetime", "amount", "currency_name", "trx_cat_ru",
        "mcc_desc", "mcc_code_desc", "label",
    ]
    return df[keep].copy()


def client_label_table(df: pd.DataFrame) -> pd.DataFrame:
    label_counts = df.groupby("customer_id")["label"].nunique()
    bad = label_counts[label_counts > 1]
    if len(bad):
        raise ValueError(f"Some clients have multiple labels: {bad.head().to_dict()}")
    return df.drop_duplicates("customer_id")[["customer_id", "label"]].reset_index(drop=True)


def split_clients(
    clients: pd.DataFrame,
    test_ids: set[int | str],
    val_size_from_remaining: float = DEFAULT_VAL_SIZE_FROM_REMAINING,
) -> tuple[set[int], set[int], set[int]]:
    client_ids = set(normalize_id_series(clients["customer_id"]).tolist())
    missing_test_ids = test_ids - client_ids
    if missing_test_ids:
        preview = sorted(list(missing_test_ids))[:10]
        raise ValueError(f"Official test ids missing from labeled clients: {len(missing_test_ids)}; preview={preview}")

    test_clients = clients[clients["customer_id"].isin(test_ids)].copy()
    remaining = clients[~clients["customer_id"].isin(test_ids)].copy()
    if remaining.empty:
        raise ValueError("No clients left after removing official test ids")

    train_clients, val_clients = train_test_split(
        remaining,
        test_size=val_size_from_remaining,
        random_state=RANDOM_STATE,
        stratify=remaining["label"],
    )
    return (
        set(train_clients["customer_id"].astype(int)),
        set(val_clients["customer_id"].astype(int)),
        set(test_clients["customer_id"].astype(int)),
    )


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    client_labels = df.drop_duplicates("customer_id")["label"]
    amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    category_counts = df["mcc_code_desc"].value_counts().head(20)
    return {
        "rows": int(len(df)),
        "clients": int(df["customer_id"].nunique()),
        "min_date": str(pd.to_datetime(df["tr_datetime"], errors="coerce").min()),
        "max_date": str(pd.to_datetime(df["tr_datetime"], errors="coerce").max()),
        "label_counts_by_client": {str(k): int(v) for k, v in client_labels.value_counts().sort_index().items()},
        "majority_accuracy": float(client_labels.value_counts(normalize=True).max()),
        "transactions_per_client_mean": float(df.groupby("customer_id").size().mean()),
        "transactions_per_client_median": float(df.groupby("customer_id").size().median()),
        "amount_sum": float(amount.sum()),
        "amount_mean": float(amount.mean()),
        "amount_median": float(amount.median()),
        "amount_min": float(amount.min()),
        "amount_max": float(amount.max()),
        "top_categories_by_rows": {str(k): int(v) for k, v in category_counts.items()},
    }


def validate_disjoint(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    ids = {
        "train": set(train["customer_id"].unique()),
        "val": set(val["customer_id"].unique()),
        "test": set(test["customer_id"].unique()),
    }
    overlap = {
        "train_val": len(ids["train"] & ids["val"]),
        "train_test": len(ids["train"] & ids["test"]),
        "val_test": len(ids["val"] & ids["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Client leakage across splits: {overlap}")
    return overlap


def write_summary_text(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"Dataset: {report['dataset']}",
        f"Split policy: {report['split_policy']}",
        f"Random state: {report['random_state']}",
        "",
    ]
    for split_name, split_report in report["splits"].items():
        lines.extend([
            f"[{split_name}]",
            f"rows: {split_report['rows']}",
            f"clients: {split_report['clients']}",
            f"date range: {split_report['min_date']} - {split_report['max_date']}",
            f"label counts: {split_report['label_counts_by_client']}",
            f"majority accuracy: {split_report['majority_accuracy']:.6f}",
            f"transactions/client mean: {split_report['transactions_per_client_mean']:.3f}",
            f"transactions/client median: {split_report['transactions_per_client_median']:.3f}",
            f"amount mean: {split_report['amount_mean']:.6f}",
            f"amount median: {split_report['amount_median']:.6f}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(
    spec: DatasetSpec,
    val_size_from_remaining: float = DEFAULT_VAL_SIZE_FROM_REMAINING,
) -> dict[str, Any]:
    print(f"\nPreparing {spec.name}...")
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    transactions_raw, labels_raw = spec.loader()
    df = spec.normalizer(transactions_raw, labels_raw)
    clients = client_label_table(df)
    test_ids = read_id_set(spec.test_ids_path, spec.test_id_column)

    train_ids, val_ids, test_ids_matched = split_clients(clients, test_ids, val_size_from_remaining)
    train = df[df["customer_id"].isin(train_ids)].copy()
    val = df[df["customer_id"].isin(val_ids)].copy()
    test = df[df["customer_id"].isin(test_ids_matched)].copy()

    overlap = validate_disjoint(train, val, test)

    train.to_csv(spec.output_dir / "train.csv", index=False)
    val.to_csv(spec.output_dir / "val.csv", index=False)
    test.to_csv(spec.output_dir / "test.csv", index=False)

    report = {
        "dataset": spec.name,
        "source_dataset": spec.dataset_name,
        "split_policy": "provided official test ids; stratified train/val over remaining labeled clients",
        "test_ids_path": str(spec.test_ids_path),
        "test_id_column": spec.test_id_column,
        "random_state": RANDOM_STATE,
        "val_size_from_remaining": val_size_from_remaining,
        "n_official_test_ids": len(test_ids),
        "n_labeled_clients": int(clients["customer_id"].nunique()),
        "client_overlap": overlap,
        "splits": {
            "train": summarize_split(train),
            "val": summarize_split(val),
            "test": summarize_split(test),
        },
        "columns": list(train.columns),
    }

    with open(spec.output_dir / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_summary_text(report, spec.output_dir / "split_summary.txt")

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        summary = report["splits"][split_name]
        print(
            f"  {split_name:<5}: rows={len(split_df):,}, "
            f"clients={summary['clients']:,}, labels={summary['label_counts_by_client']}, "
            f"majority={summary['majority_accuracy']:.4f}"
        )
    print(f"  report: {spec.output_dir / 'split_report.json'}")
    print(f"  summary: {spec.output_dir / 'split_summary.txt'}")
    return report


def build_specs(test_ids_dir: Path = Path("data/test_ids"), output_root: Path = Path("data")) -> dict[str, DatasetSpec]:
    return {
        "gender": DatasetSpec(
            name="gender",
            dataset_name="pytorch-lifestream/transactions-gender",
            output_dir=output_root / "gender",
            test_ids_path=test_ids_dir / "gender_test_ids.csv",
            test_id_column="customer_id",
            label_column="label",
            loader=load_gender_raw,
            normalizer=normalize_gender,
        ),
        "age": DatasetSpec(
            name="age",
            dataset_name="pytorch-lifestream/age-group-prediction",
            output_dir=output_root / "age",
            test_ids_path=test_ids_dir / "age_test_ids.csv",
            test_id_column="client_id",
            label_column="label",
            loader=load_age_raw,
            normalizer=normalize_age,
        ),
        "rosbank": DatasetSpec(
            name="rosbank",
            dataset_name="pytorch-lifestream/rosbank-churn",
            output_dir=output_root / "rosbank",
            test_ids_path=test_ids_dir / "rosbank_test_ids.csv",
            test_id_column="cl_id",
            label_column="label",
            loader=load_rosbank_raw,
            normalizer=normalize_rosbank,
        ),
    }

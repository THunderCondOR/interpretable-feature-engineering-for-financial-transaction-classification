"""Download, normalize, split, and summarize benchmark datasets.

Each dataset is prepared with the fixed test client ids stored in data/test_ids.
The prepared CSV files use the internal schema consumed by the rest of the
pipeline:

    customer_id, tr_datetime, amount, mcc_code_desc, label

Outputs:
    data/<dataset>/train.csv
    data/<dataset>/val.csv
    data/<dataset>/test.csv
    data/<dataset>/split_report.json
    data/<dataset>/split_summary.txt
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
VAL_SIZE_FROM_REMAINING = 1.0 / 9.0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    output_dir: Path
    test_ids_path: Path
    test_id_column: str
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


def normalize_ids(values: pd.Series) -> pd.Series:
    return values.dropna().astype("int64")


def load_test_ids(path: Path, column: str) -> set[int]:
    return set(normalize_ids(pd.read_csv(path)[column]).tolist())


def parse_rosbank_datetime(value: Any) -> pd.Timestamp:
    months = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
        "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    match = re.match(r"^(\d{2})([A-Z]{3})(\d{2}):(\d{2}:\d{2}:\d{2})$", str(value))
    day, month, year, time = match.groups()
    return pd.to_datetime(f"20{year}-{months[month]}-{day} {time}")


def mcc_to_text(value: Any) -> str:
    return MCC_TO_DESC.get(int(value), f"MCC {int(value)}")


def load_gender_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = load_dataset("pytorch-lifestream/transactions-gender", "transactions_data", split="train").to_pandas()
    labels = load_dataset("pytorch-lifestream/transactions-gender", "labels", split="train").to_pandas()
    return transactions, labels


def normalize_gender(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = transactions.merge(labels[["customer_id", "gender"]], on="customer_id", how="inner")
    df = df.rename(columns={"gender": "label"})
    df["mcc_code_desc"] = df["mcc_code"].apply(mcc_to_text)
    df["tr_datetime"] = pd.to_datetime(df["tr_datetime"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"])
    df["label"] = df["label"].astype(int)
    columns = ["customer_id", "tr_datetime", "amount", "mcc_code_desc", "label", "mcc_code", "tr_type", "term_id"]
    return df[columns].copy()


def load_age_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = load_dataset("pytorch-lifestream/age-group-prediction", "transactions_train", split="train").to_pandas()
    labels = load_dataset("pytorch-lifestream/age-group-prediction", "train_target", split="train").to_pandas()
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
    df["tr_datetime"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["tr_datetime"].astype(int), unit="D")
    df["mcc_code_desc"] = df["mcc_code_desc"].map(lambda value: f"operation group {value}")
    df["amount"] = pd.to_numeric(df["amount"])
    df["label"] = df["label"].astype(int)
    return df[["customer_id", "tr_datetime", "amount", "mcc_code_desc", "label"]].copy()


def load_rosbank_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = load_dataset("pytorch-lifestream/rosbank-churn", "train", split="train").to_pandas()
    labels = transactions[["cl_id", "target_flag"]].drop_duplicates("cl_id")
    return transactions, labels


def normalize_rosbank(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["tr_datetime"] = df["TRDATETIME"].apply(parse_rosbank_datetime)
    df["currency_name"] = df["currency"].map(lambda value: CURRENCY_MAP[int(value)])
    df["trx_cat_ru"] = df["trx_category"].map(lambda value: TRX_CATEGORY_MAP[value])
    df["mcc_desc"] = df["MCC"].apply(mcc_to_text)
    df["mcc_code_desc"] = df["mcc_desc"] + " [" + df["trx_cat_ru"] + "]"
    df = df.rename(columns={"cl_id": "customer_id", "target_flag": "label"})
    df["customer_id"] = df["customer_id"].astype(int)
    df["amount"] = pd.to_numeric(df["amount"])
    df["label"] = df["label"].astype(int)
    columns = ["customer_id", "tr_datetime", "amount", "currency_name", "trx_cat_ru", "mcc_desc", "mcc_code_desc", "label"]
    return df[columns].copy()


def client_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("customer_id")[["customer_id", "label"]].reset_index(drop=True)


def split_clients(clients: pd.DataFrame, test_ids: set[int], val_size: float) -> tuple[set[int], set[int], set[int]]:
    test_clients = clients[clients["customer_id"].isin(test_ids)]
    remaining = clients[~clients["customer_id"].isin(test_ids)]
    train_clients, val_clients = train_test_split(
        remaining,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=remaining["label"],
    )
    return (
        set(train_clients["customer_id"].astype(int)),
        set(val_clients["customer_id"].astype(int)),
        set(test_clients["customer_id"].astype(int)),
    )


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    labels = df.drop_duplicates("customer_id")["label"]
    txn_per_client = df.groupby("customer_id").size()
    amount = pd.to_numeric(df["amount"])
    return {
        "rows": int(len(df)),
        "clients": int(df["customer_id"].nunique()),
        "min_date": str(pd.to_datetime(df["tr_datetime"]).min()),
        "max_date": str(pd.to_datetime(df["tr_datetime"]).max()),
        "label_counts_by_client": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "majority_accuracy": float(labels.value_counts(normalize=True).max()),
        "transactions_per_client_mean": float(txn_per_client.mean()),
        "transactions_per_client_median": float(txn_per_client.median()),
        "amount_sum": float(amount.sum()),
        "amount_mean": float(amount.mean()),
        "amount_median": float(amount.median()),
        "amount_min": float(amount.min()),
        "amount_max": float(amount.max()),
        "top_categories_by_rows": {str(k): int(v) for k, v in df["mcc_code_desc"].value_counts().head(20).items()},
    }


def check_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, expected_test_ids: set[int]) -> dict[str, int]:
    train_ids = set(train["customer_id"].unique())
    val_ids = set(val["customer_id"].unique())
    test_ids = set(test["customer_id"].unique())
    overlap = {
        "train_val": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "val_test": len(val_ids & test_ids),
    }
    assert overlap == {"train_val": 0, "train_test": 0, "val_test": 0}, overlap
    assert test_ids == expected_test_ids, (len(expected_test_ids - test_ids), len(test_ids - expected_test_ids))
    return overlap


def write_summary(report: dict[str, Any], path: Path) -> None:
    lines = [f"Dataset: {report['dataset']}", f"Source: {report['source']}", ""]
    for split_name, split_report in report["splits"].items():
        lines.extend(
            [
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
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(spec: DatasetSpec, val_size: float = VAL_SIZE_FROM_REMAINING) -> dict[str, Any]:
    print(f"\nPreparing {spec.name}")
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    transactions, labels = spec.loader()
    data = spec.normalizer(transactions, labels)
    labels_by_client = client_labels(data)
    test_ids = load_test_ids(spec.test_ids_path, spec.test_id_column)

    train_ids, val_ids, test_ids = split_clients(labels_by_client, test_ids, val_size)
    train = data[data["customer_id"].isin(train_ids)].copy()
    val = data[data["customer_id"].isin(val_ids)].copy()
    test = data[data["customer_id"].isin(test_ids)].copy()
    overlap = check_splits(train, val, test, test_ids)

    train.to_csv(spec.output_dir / "train.csv", index=False)
    val.to_csv(spec.output_dir / "val.csv", index=False)
    test.to_csv(spec.output_dir / "test.csv", index=False)

    report = {
        "dataset": spec.name,
        "source": spec.source,
        "test_ids_path": str(spec.test_ids_path),
        "test_id_column": spec.test_id_column,
        "random_state": RANDOM_STATE,
        "val_size_from_remaining": val_size,
        "client_overlap": overlap,
        "splits": {
            "train": summarize_split(train),
            "val": summarize_split(val),
            "test": summarize_split(test),
        },
        "columns": list(train.columns),
    }

    with open(spec.output_dir / "split_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    write_summary(report, spec.output_dir / "split_summary.txt")

    for split_name in ["train", "val", "test"]:
        split_report = report["splits"][split_name]
        print(
            f"  {split_name:<5}: rows={split_report['rows']:,}, "
            f"clients={split_report['clients']:,}, "
            f"labels={split_report['label_counts_by_client']}, "
            f"majority={split_report['majority_accuracy']:.4f}"
        )
    return report


def build_specs(test_ids_dir: Path = Path("data/test_ids"), data_dir: Path = Path("data")) -> dict[str, DatasetSpec]:
    return {
        "gender": DatasetSpec(
            name="gender",
            source="pytorch-lifestream/transactions-gender",
            output_dir=data_dir / "gender",
            test_ids_path=test_ids_dir / "gender_test_ids.csv",
            test_id_column="customer_id",
            loader=load_gender_raw,
            normalizer=normalize_gender,
        ),
        "age": DatasetSpec(
            name="age",
            source="pytorch-lifestream/age-group-prediction",
            output_dir=data_dir / "age",
            test_ids_path=test_ids_dir / "age_test_ids.csv",
            test_id_column="client_id",
            loader=load_age_raw,
            normalizer=normalize_age,
        ),
        "rosbank": DatasetSpec(
            name="rosbank",
            source="pytorch-lifestream/rosbank-churn",
            output_dir=data_dir / "rosbank",
            test_ids_path=test_ids_dir / "rosbank_test_ids.csv",
            test_id_column="cl_id",
            loader=load_rosbank_raw,
            normalizer=normalize_rosbank,
        ),
    }

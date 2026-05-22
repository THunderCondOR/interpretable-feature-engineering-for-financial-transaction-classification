"""
prepare_rosbank.py

Downloads the labeled train subset of pytorch-lifestream/rosbank-churn,
normalizes it to the repository internal schema, and creates local
train/val/test splits by customer_id.

Important design choice:
- The HuggingFace test subset does not contain target_flag.
- Therefore it is an inference-only split and cannot be used for metrics.
- For experiments with metrics we split the labeled authors' train subset:
    local train: 70% of clients
    local val:   10% of clients
    local test:  20% of clients
- Splitting is stratified by customer-level label.

Outputs:
    data/rosbank/train.csv
    data/rosbank/val.csv
    data/rosbank/test.csv
    data/rosbank/split_report.json

Run:
    python prepare_rosbank.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data/rosbank")
DATASET_NAME = "pytorch-lifestream/rosbank-churn"
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.10
TEST_SIZE = 0.20

CURRENCY_MAP = {
    810: "Рубль", 643: "Рубль", 978: "Евро", 840: "Доллар США",
    826: "Фунт стерлингов", 756: "Швейцарский франк", 156: "Китайский юань",
    392: "Японская иена", 980: "Украинская гривна", 398: "Казахстанский тенге",
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

# Compact hand-built MCC map. Unknown codes are kept as "Категория <code>".
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


def parse_trdatetime(value: Any) -> pd.Timestamp:
    """Convert strings like '21OCT17:13:08:28' to pandas Timestamp."""
    if pd.isna(value):
        return pd.NaT
    s = str(value)
    months = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
        "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2}):(\d{2}:\d{2}:\d{2})$", s)
    if m:
        day, mon, yy, time = m.groups()
        return pd.to_datetime(f"20{yy}-{months.get(mon, '01')}-{day} {time}", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _load_labeled_train_subset() -> pd.DataFrame:
    """Load the labeled authors' train subset with a robust fallback."""
    try:
        return load_dataset(DATASET_NAME, "train", split="train").to_pandas()
    except Exception as first_error:
        try:
            return load_dataset(DATASET_NAME, split="train").to_pandas()
        except Exception:
            raise first_error


def _mcc_to_text(mcc: Any) -> str:
    try:
        return MCC_TO_DESC.get(int(mcc), f"Категория {int(mcc)}")
    except Exception:
        return f"Категория {mcc}"


def normalize_rosbank(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw Rosbank columns to the internal schema."""
    required = {"cl_id", "MCC", "currency", "TRDATETIME", "amount", "trx_category", "target_flag"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}; columns={list(raw.columns)}")

    df = raw.copy()
    df["tr_datetime"] = df["TRDATETIME"].apply(parse_trdatetime)
    df["currency_name"] = df["currency"].map(
        lambda x: CURRENCY_MAP.get(int(x), f"Валюта {x}") if pd.notna(x) else "неизвестная валюта"
    )
    df["trx_cat_ru"] = df["trx_category"].map(TRX_CATEGORY_MAP).fillna(df["trx_category"].astype(str))
    df["mcc_desc"] = df["MCC"].apply(_mcc_to_text)
    df["mcc_code_desc"] = df["mcc_desc"] + " [" + df["trx_cat_ru"] + "]"
    df = df.rename(columns={"cl_id": "customer_id", "target_flag": "label"})

    keep = [
        "customer_id", "tr_datetime", "amount", "currency_name", "trx_cat_ru",
        "mcc_desc", "mcc_code_desc", "label",
    ]
    result = df[keep].copy()
    result["customer_id"] = result["customer_id"].astype(int)
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce").fillna(0.0)
    result["tr_datetime"] = pd.to_datetime(result["tr_datetime"], errors="coerce")
    result["label"] = result["label"].astype(int)
    return result


def client_label_table(df: pd.DataFrame) -> pd.DataFrame:
    label_counts = df.groupby("customer_id")["label"].nunique()
    bad = label_counts[label_counts > 1]
    if len(bad):
        raise ValueError(f"Some clients have multiple labels: {bad.head().to_dict()}")
    return df.drop_duplicates("customer_id")[["customer_id", "label"]].reset_index(drop=True)


def split_labeled_clients(clients: pd.DataFrame) -> tuple[set[int], set[int], set[int]]:
    """Create 70/10/20 stratified split by client labels."""
    train_clients, temp_clients = train_test_split(
        clients,
        test_size=VAL_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=clients["label"],
    )
    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_clients, test_clients = train_test_split(
        temp_clients,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=temp_clients["label"],
    )
    return (
        set(train_clients["customer_id"].astype(int)),
        set(val_clients["customer_id"].astype(int)),
        set(test_clients["customer_id"].astype(int)),
    )


def summarize_split(df: pd.DataFrame) -> dict:
    client_labels = df.drop_duplicates("customer_id")["label"]
    label_counts = client_labels.value_counts(dropna=False).sort_index().to_dict()
    return {
        "rows": int(len(df)),
        "clients": int(df["customer_id"].nunique()),
        "min_date": str(pd.to_datetime(df["tr_datetime"], errors="coerce").min()),
        "max_date": str(pd.to_datetime(df["tr_datetime"], errors="coerce").max()),
        "label_counts_by_client": {str(k): int(v) for k, v in label_counts.items()},
        "positive_rate_by_client": float(client_labels.mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading labeled authors' train subset...")
    raw = _load_labeled_train_subset()
    print(f"  raw rows={len(raw):,}, clients={raw['cl_id'].nunique():,}, columns={list(raw.columns)}")

    df = normalize_rosbank(raw)
    clients = client_label_table(df)
    print("Client-level label distribution:")
    print(clients["label"].value_counts().sort_index().to_string())

    train_ids, val_ids, test_ids = split_labeled_clients(clients)
    train = df[df["customer_id"].isin(train_ids)].copy()
    val = df[df["customer_id"].isin(val_ids)].copy()
    test = df[df["customer_id"].isin(test_ids)].copy()

    # Ensure disjoint client ids.
    ids = {"train": set(train.customer_id), "val": set(val.customer_id), "test": set(test.customer_id)}
    overlap_report = {
        "train_val": len(ids["train"] & ids["val"]),
        "train_test": len(ids["train"] & ids["test"]),
        "val_test": len(ids["val"] & ids["test"]),
    }
    if any(overlap_report.values()):
        raise RuntimeError(f"Client leakage across splits: {overlap_report}")

    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    report = {
        "dataset": DATASET_NAME,
        "split_policy": "authors labeled train subset -> stratified local 70/10/20 split by customer_id",
        "random_state": RANDOM_STATE,
        "fractions_by_clients": {"train": TRAIN_SIZE, "val": VAL_SIZE, "test": TEST_SIZE},
        "splits": {
            "train": summarize_split(train),
            "val": summarize_split(val),
            "test": summarize_split(test),
        },
        "client_overlap": overlap_report,
        "columns": list(train.columns),
    }
    with open(OUT_DIR / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Saved:")
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        s = summarize_split(split_df)
        print(
            f"  {name:<5}: rows={s['rows']:,}, clients={s['clients']:,}, "
            f"labels={s['label_counts_by_client']}, positive_rate={s['positive_rate_by_client']:.4f}"
        )
    print(f"  report: {OUT_DIR / 'split_report.json'}")


if __name__ == "__main__":
    main()

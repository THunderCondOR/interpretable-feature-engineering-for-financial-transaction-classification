"""
prepare_rosbank.py

Скачивает rosbank-churn, делает полную предобработку и сохраняет сплиты.

Схема датасета (из HuggingFace viewer):
    PERIOD        string   — месяц транзакции (01/10/2017)
    cl_id         int64    — ID клиента
    MCC           int64    — 4-значный международный MCC код
    channel_type  string   — канал (почти всегда null)
    currency      int64    — код валюты ISO 4217
    TRDATETIME    string   — дата/время (21OCT17:00:00:00)
    amount        float64  — сумма (всегда положительная)
    trx_category  string   — тип операции (POS, DEPOSIT, C2C_OUT, WD_ATM_*, ...)
    target_flag   int64    — таргет: 1=отток, 0=активный
    target_sum    float64  — вспомогательная колонка (не используется)

Результат: data/rosbank/train.csv, val.csv, test.csv

Итоговые колонки:
    customer_id   — ID клиента (int)
    tr_datetime   — дата/время в ISO формате
    amount        — сумма транзакции
    currency_name — название валюты (Рубль, Евро, ...)
    trx_cat_ru    — тип операции на русском
    mcc_code_desc — MCC описание + тип операции
    label         — 0=активный, 1=отток

Запуск:
    python prepare_rosbank.py
"""

import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path


# ---------------------------------------------------------------------------
# Валюты ISO 4217 → название
# ---------------------------------------------------------------------------

CURRENCY_MAP = {
    810: "Рубль",       # старый код RUR (встречается в данных)
    643: "Рубль",       # ISO RUB
    978: "Евро",
    840: "Доллар США",
    826: "Фунт стерлингов",
    756: "Швейцарский франк",
    156: "Китайский юань",
    392: "Японская иена",
    980: "Украинская гривна",
    398: "Казахстанский тенге",
}


# ---------------------------------------------------------------------------
# Тип операции → русское описание
# ---------------------------------------------------------------------------

TRX_CATEGORY_MAP = {
    "POS":              "оплата картой",
    "WD_ATM_PARTNER":   "снятие наличных (банкомат партнёра)",
    "WD_ATM_ROS":       "снятие наличных (банкомат Росбанка)",
    "WD_ATM_OTHER":     "снятие наличных (другой банк)",
    "CAT":              "снятие наличных через кассу",
    "DEPOSIT":          "пополнение счёта",
    "C2C_OUT":          "перевод на карту",
    "C2C_IN":           "входящий перевод с карты",
    "MBO":              "мобильный банк",
    "WEB":              "интернет-банк",
    "BACK":             "возврат средств",
}


# ---------------------------------------------------------------------------
# MCC коды → русское описание
# ---------------------------------------------------------------------------

MCC_TO_DESC = {
    5200: "Строительные материалы",
    5251: "Скобяные товары",
    5261: "Садовые принадлежности",
    5310: "Универсальные магазины",
    5311: "Универмаги",
    5331: "Дисконт-магазины",
    5411: "Супермаркеты",
    5412: "Продуктовые магазины",
    5441: "Кондитерские",
    5462: "Булочные",
    5499: "Продовольственные магазины",
    5511: "Автосалоны",
    5531: "Автозапчасти",
    5541: "АЗС",
    5542: "АЗС (автомат)",
    5611: "Мужская одежда",
    5621: "Женская одежда",
    5631: "Женские аксессуары",
    5641: "Детская одежда",
    5651: "Одежда",
    5661: "Обувь",
    5691: "Одежда (разное)",
    5699: "Одежда и аксессуары",
    5712: "Мебель",
    5719: "Товары для дома",
    5722: "Бытовая техника",
    5732: "Электроника",
    5734: "Компьютеры и ПО",
    5812: "Рестораны",
    5813: "Бары и клубы",
    5814: "Фастфуд",
    5912: "Аптеки",
    5921: "Алкоголь",
    5941: "Спорттовары",
    5944: "Ювелирные изделия",
    5945: "Игрушки",
    5947: "Подарки и сувениры",
    5977: "Косметика",
    5992: "Цветы",
    5122: "Лекарства",
    6010: "Снятие наличных (касса)",
    6011: "Снятие наличных (банкомат)",
    6012: "Финансовые услуги",
    6051: "Денежные переводы",
    4111: "Городской транспорт",
    4112: "Железная дорога",
    4121: "Такси",
    4131: "Автобус",
    4511: "Авиа",
    4722: "Туристические агентства",
    4816: "Интернет-сервисы",
    4829: "Денежные переводы",
    4900: "ЖКХ",
    7011: "Отели",
    7230: "Салоны красоты",
    7298: "Фитнес",
    7512: "Аренда авто",
    7523: "Парковки",
    7531: "Автосервис",
    7534: "Шины",
    7542: "Автомойки",
    7832: "Кинотеатры",
    7999: "Развлечения",
    8011: "Врачи",
    8021: "Стоматология",
    8099: "Медицина",
    8299: "Образование",
}


def parse_trdatetime(s: str) -> str:
    """Конвертирует '21OCT17:00:00:00' → '2017-10-21 00:00:00'."""
    months = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    try:
        m = re.match(r"(\d{2})([A-Z]{3})(\d{2}):(\d{2}:\d{2}:\d{2})", str(s))
        if m:
            day, mon, yr, time = m.groups()
            return f"20{yr}-{months[mon]}-{day} {time}"
    except Exception:
        pass
    return str(s)


def main():
    Path("data/rosbank").mkdir(parents=True, exist_ok=True)

    # ── Загрузить ────────────────────────────────────────────────────────────
    print("Загружаем pytorch-lifestream/rosbank-churn...")
    df = load_dataset("pytorch-lifestream/rosbank-churn", split="train").to_pandas()
    print(f"  Строк: {len(df):,}, клиентов: {df['cl_id'].nunique()}")
    print(f"  Колонки: {list(df.columns)}")
    print(f"\n  target_flag (по клиентам):")
    print(df.drop_duplicates("cl_id")["target_flag"].value_counts().to_string())
    print(f"\n  currency распределение:")
    print(df["currency"].value_counts().to_string())
    print(f"\n  trx_category распределение:")
    print(df["trx_category"].value_counts().to_string())

    # ── Дата/время ───────────────────────────────────────────────────────────
    print("\nПарсим TRDATETIME...")
    df["tr_datetime"] = df["TRDATETIME"].apply(parse_trdatetime)

    # ── Валюта → название ────────────────────────────────────────────────────
    df["currency_name"] = df["currency"].map(
        lambda x: CURRENCY_MAP.get(int(x), f"Валюта {x}") if pd.notna(x) else "Рубль"
    )
    print(f"\n  currency_name распределение:")
    print(df["currency_name"].value_counts().to_string())

    # ── Тип операции → русский ───────────────────────────────────────────────
    df["trx_cat_ru"] = df["trx_category"].map(TRX_CATEGORY_MAP).fillna(df["trx_category"])

    # ── MCC → текст + тип операции ───────────────────────────────────────────
    def mcc_to_text(mcc):
        try:
            return MCC_TO_DESC.get(int(mcc), f"Категория {int(mcc)}")
        except (ValueError, TypeError):
            return f"Категория {mcc}"

    df["mcc_code_desc"] = (
        df["MCC"].apply(mcc_to_text) + " [" + df["trx_cat_ru"] + "]"
    )

    # Покрытие маппинга MCC
    unmapped = df[df["mcc_code_desc"].str.startswith("Категория ")]["MCC"].nunique()
    total    = df["MCC"].nunique()
    print(f"\n  MCC покрытие: {total - unmapped}/{total} кодов замаппировано "
          f"({100*(total-unmapped)/total:.1f}%)")
    if unmapped > 0:
        top = (df[df["mcc_code_desc"].str.startswith("Категория ")]
               ["MCC"].value_counts().head(10))
        print(f"  Топ незамаппированных:\n{top.to_string()}")

    # ── Финальный DataFrame ──────────────────────────────────────────────────
    df = df.rename(columns={"cl_id": "customer_id", "target_flag": "label"})
    keep = ["customer_id", "tr_datetime", "amount",
            "currency_name", "trx_cat_ru", "mcc_code_desc", "label"]
    df = df[keep]
    df["label"] = df["label"].astype(int)

    print(f"\n  Итого: {len(df):,} строк, {df['customer_id'].nunique()} клиентов")
    print(f"  Итоговые колонки: {list(df.columns)}")

    # ── Сплит 70 / 10 / 20 ───────────────────────────────────────────────────
    clients = df["customer_id"].unique()
    train_ids, temp     = train_test_split(clients, test_size=0.30,  random_state=42)
    val_ids,   test_ids = train_test_split(temp,    test_size=0.667, random_state=42)

    df[df["customer_id"].isin(train_ids)].to_csv("data/rosbank/train.csv", index=False)
    df[df["customer_id"].isin(val_ids)  ].to_csv("data/rosbank/val.csv",   index=False)
    df[df["customer_id"].isin(test_ids) ].to_csv("data/rosbank/test.csv",  index=False)

    print(f"\n  Сохранено в data/rosbank/:")
    print(f"    train: {len(train_ids)} клиентов")
    print(f"    val:   {len(val_ids)} клиентов")
    print(f"    test:  {len(test_ids)} клиентов")


if __name__ == "__main__":
    main()
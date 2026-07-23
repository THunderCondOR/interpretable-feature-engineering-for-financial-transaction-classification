import pytest

from src.data.mcc import MCC_TO_DESC, MCC_TO_DESC_EN
from src.data.prompt_locale import (
    assert_english_model_text,
    english_category,
    english_currency,
    english_operation_type,
)


def test_every_known_mcc_has_a_deterministic_english_name():
    assert set(MCC_TO_DESC_EN) == set(MCC_TO_DESC)
    # Prepared splits retain the display value, not the MCC code. A few MCC
    # codes share the same Russian display value, so that value necessarily
    # has one deterministic English rendering at the prompt boundary.
    rendered = {}
    for russian in set(MCC_TO_DESC.values()):
        translated = english_category(russian)
        assert translated
        assert_english_model_text(translated, context=russian)
        rendered[russian] = translated
    assert len(rendered) == len(set(MCC_TO_DESC.values()))


def test_rosbank_composites_operation_types_and_currency_codes_translate():
    assert english_category(
        "Супермаркеты [оплата картой]"
    ).endswith("[card payment]")
    assert english_operation_type("пополнение счета") == "account deposit"
    assert english_currency("Валюта 978") == "Currency code 978"


def test_unknown_cyrillic_model_text_fails_closed():
    with pytest.raises(ValueError, match="Unknown Russian category"):
        english_category("Неизвестная категория")
    with pytest.raises(ValueError, match="Cyrillic leaked"):
        assert_english_model_text("mixed текст", context="test")

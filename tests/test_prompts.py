from pathlib import Path


ROOT = Path(__file__).parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8").lower()


def test_gender_prompts_do_not_prime_external_stereotypes():
    combined = "\n".join([
        _read("prompts/gender/explanation_generation/system_prompt.txt"),
        _read("prompts/gender/explanation_generation/user_prompt.txt"),
        _read("prompts/gender/claims_extraction/user_prompt.txt"),
    ])
    for priming in ("косметик", "цветы", "автозапчаст", "детск", "ювелир"):
        assert priming not in combined
    assert "training split" in combined
    assert "top-k" in combined


def test_age_prompts_do_not_supply_age_archetypes_or_call_amount_income():
    combined = "\n".join([
        _read("prompts/age/explanation_generation/system_prompt.txt"),
        _read("prompts/age/explanation_generation/user_prompt.txt"),
    ])
    for priming in ("фастфуд", "ипотек", "образование детей", "типичные поведенческие"):
        assert priming not in combined
    assert "amount обозначает величину операции, а не доход" in combined
    assert "training split" in combined


def test_user_prompts_explicitly_scope_summaries_to_training_split():
    for dataset in ("gender", "age", "rosbank"):
        text = _read(f"prompts/{dataset}/explanation_generation/user_prompt.txt")
        assert "training split" in text
        assert "по всему датасету" not in text


def test_reasoning_guide_is_flexible_not_a_fixed_template():
    for dataset in ("gender", "age", "rosbank"):
        text = _read(f"prompts/{dataset}/explanation_generation/system_prompt.txt")
        assert "структуру и набор рассмотренных признаков выбери самостоятельно" in text
        assert "evidence —" not in text
        assert "interpretation —" not in text
        assert "5–10" not in text
        assert "сначала обязательно" not in text

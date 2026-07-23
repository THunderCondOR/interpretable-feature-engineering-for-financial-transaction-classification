from pathlib import Path

import yaml

from src.experiments.config_builder import build_runtime_config
from src.pipeline.prompt_builder import validate_prompt_contract


ROOT = Path(__file__).parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8").lower()


def _config(dataset: str, semantics: str | None = None) -> dict:
    with (ROOT / "configs" / f"{dataset}.yaml").open(encoding="utf-8") as file:
        base = yaml.safe_load(file)
    with (ROOT / "configs/v2/qwen.yaml").open(encoding="utf-8") as file:
        model = yaml.safe_load(file)
    return build_runtime_config(
        base,
        model,
        run_id="prompt-test",
        variant="guided_zero_shot_v4",
        label_semantics=semantics,
    )


def test_common_prompts_are_english_and_do_not_prime_stereotypes():
    combined = "\n".join(
        [
            _read("prompts/common/explanation_generation/system_prompt.txt"),
            _read("prompts/common/explanation_generation/user_prompt.txt"),
            _read("prompts/common/claims_extraction/system_prompt.txt"),
            _read("prompts/common/claims_extraction/user_prompt.txt"),
        ]
    )
    for priming in (
        "cosmetics imply",
        "flowers imply",
        "car accessories imply",
        "child-related purchases imply",
        "women usually",
        "men usually",
    ):
        assert priming not in combined
    assert "training split" in combined
    assert "top-k" in combined
    assert "in english" in combined


def test_reasoning_guide_is_evidence_bounded_but_not_a_fixed_template():
    text = _read("prompts/common/explanation_generation/system_prompt.txt")
    assert "reason freely" in text
    assert "choose your own reasoning structure" in text
    assert "interpretations as hypotheses" in text
    assert "do not invent transactions" in text
    for rigid_instruction in (
        "evidence:",
        "interpretation:",
        "exactly five",
        "first, you must",
        "mention every statistic",
    ):
        assert rigid_instruction not in text


def test_all_dataset_prompt_contracts_are_valid_english():
    for dataset in ("gender", "rosbank"):
        validate_prompt_contract(_config(dataset))
    for semantics in ("age_opaque", "age_ordered"):
        validate_prompt_contract(_config("age", semantics))


def test_age_opaque_hides_order_and_boundaries_while_allowing_hypotheses():
    config = _config("age", "age_opaque")
    guidance = config["dataset"]["prompt_dataset_guidance"].lower()
    assert "order are not provided" in guidance
    assert "exact numerical boundaries" in guidance
    assert "student-like" in guidance
    assert "only when supported" in guidance
    for boundary in ("18-25", "26-35", "36-45", "46+"):
        assert boundary not in guidance


def test_age_ordered_exposes_only_relative_order_not_numeric_boundaries():
    config = _config("age", "age_ordered")
    guidance = config["dataset"]["prompt_dataset_guidance"].lower()
    assert "age_group_a is the youngest" in guidance
    assert "age_group_d is the oldest" in guidance
    assert "exact numerical boundaries are not provided" in guidance
    for boundary in ("18-25", "26-35", "36-45", "46+"):
        assert boundary not in guidance

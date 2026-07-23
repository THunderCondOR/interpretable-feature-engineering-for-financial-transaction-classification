from types import SimpleNamespace

from src.pipeline.claims_extractor import (
    _behavioral_text,
    _parse_claim_result,
    _validate_claims,
)


def _result(content: str, finish_reason: str = "stop") -> dict:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return {
        "response": SimpleNamespace(choices=[choice]),
        "error": None,
    }


def test_final_label_is_removed_before_claim_extraction():
    rationale = (
        "The client transacts frequently across diverse categories. "
        "Activity remains stable over the observed period.\n"
        "Final: \\boxed{age_group_B}"
    )
    behavioral = _behavioral_text(rationale)
    assert "age_group_B" not in behavioral
    assert "Final:" not in behavioral


def test_claim_contract_accepts_english_qualitative_atomic_claims():
    claims, error_type, error = _validate_claims(
        [
            "The client transacts frequently across diverse categories.",
            "The client's activity may be consistent with student-like behavior.",
            "The client frequently uses operation group 14.",
        ],
        forbidden_labels={"age_group_A", "age_group_B"},
    )
    assert len(claims) == 3
    assert error_type is None
    assert error is None


def test_claim_contract_rejects_target_labels_numbers_cyrillic_and_bad_subjects():
    cases = (
        ("The client resembles age_group_A.", "TargetLabelLeakage"),
        ("The client made 12 transactions.", "NumericClaim"),
        ("The client часто uses grocery stores.", "NonEnglishClaims"),
        ("This customer transacts frequently.", "InvalidClaimSubject"),
    )
    for claim, expected in cases:
        values, error_type, _ = _validate_claims(
            [claim],
            forbidden_labels={"age_group_A"},
        )
        assert values == []
        assert error_type == expected


def test_claim_parser_rejects_truncation_and_non_string_schema():
    claims, error_type, _ = _parse_claim_result(
        _result('["The client is active."]', finish_reason="length")
    )
    assert claims == []
    assert error_type == "IncompleteResponse"

    claims, error_type, _ = _parse_claim_result(_result('["valid", 2]'))
    assert claims == []
    assert error_type == "InvalidClaimsSchema"

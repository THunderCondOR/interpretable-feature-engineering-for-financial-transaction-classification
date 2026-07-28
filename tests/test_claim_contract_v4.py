import json
from types import SimpleNamespace

from src.pipeline.claims_extractor import (
    _behavioral_text,
    _parse_claim_result,
    _validate_claims,
    run_claims_extraction,
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


def test_claim_contract_discards_invalid_items_but_keeps_valid_atomic_claims():
    values, error_type, error = _validate_claims(
        [
            "The client resembles the retained class.",
            "The client has sustained transaction activity.",
            "The client made 12 transactions.",
        ],
        forbidden_labels={"retained"},
    )
    assert values == ["The client has sustained transaction activity."]
    assert error_type is None
    assert error is None


def test_claim_parser_rejects_truncation_and_non_string_schema():
    claims, error_type, _ = _parse_claim_result(
        _result('["The client is active."]', finish_reason="length")
    )
    assert claims == []
    assert error_type == "IncompleteResponse"

    claims, error_type, _ = _parse_claim_result(_result('["valid", 2]'))
    assert claims == []
    assert error_type == "InvalidClaimsSchema"


def test_claims_defer_bad_content_without_replaying_good_requests(
    tmp_path, monkeypatch
):
    explanations = tmp_path / "explanations.jsonl"
    claims = tmp_path / "claims.jsonl"
    explanations.write_text(
        json.dumps(
            {
                "customer_id": 1,
                "label": 0,
                "label_name": "female",
                "sample_id": 0,
                "explanation": (
                    "The client maintains regular transaction activity.\n"
                    "Final: \\boxed{female}"
                ),
                "prompt_hash": "prompt",
                "client_stats_hash": "client",
                "summary_stats_hash": "summary",
                "generation_signature": "explanation",
                "min_behavioral_explanation_chars": 1,
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    call_sizes = []

    async def fake_batched_query(
        dialogues, model, llm_config, *, on_batch_complete
    ):
        call_sizes.append(len(dialogues))
        result = (
            _result("[]")
            if len(call_sizes) == 1
            else _result(
                '["The client maintains regular transaction activity."]'
            )
        )
        on_batch_complete([(0, result)])
        return [result]

    monkeypatch.setattr(
        "src.pipeline.claims_extractor.batched_query",
        fake_batched_query,
    )
    config = {
        "llm": {"default_model": "test"},
        "execution": {"until_complete": True},
        "claims_generation": {"model": "test", "max_tokens": 128},
        "experiment": {"run_id": "test", "model_slug": "test"},
        "dataset": {
            "name": "gender",
            "label_names": {"0": "female", "1": "male"},
            "claim_forbidden_terms": [],
        },
        "pipeline": {"n_claims_samples": 1},
        "prompts": {
            "base_dir": ".",
            "claims_system": "prompts/common/claims_extraction/system_prompt.txt",
            "claims_user": "prompts/common/claims_extraction/user_prompt.txt",
        },
        "output": {
            "base_dir": str(tmp_path),
            "explanations": "explanations.jsonl",
            "claims": "claims.jsonl",
        },
    }

    run_claims_extraction(
        config,
        input_path=explanations,
        output_path=claims,
    )

    assert call_sizes == [1, 1]
    record = json.loads(claims.read_text(encoding="utf-8"))
    assert record["claims"] == [
        "The client maintains regular transaction activity."
    ]
    stats = json.loads(
        claims.with_suffix(".generation_stats.json").read_text(
            encoding="utf-8"
        )
    )
    assert stats["content_validation"]["error_types"] == {"EmptyClaims": 1}
    assert stats["content_validation"]["unique_rejected_requests"] == 1


def test_claims_empty_response_stops_after_configured_attempts(
    tmp_path, monkeypatch
):
    explanations = tmp_path / "explanations.jsonl"
    claims = tmp_path / "claims.jsonl"
    explanations.write_text(
        json.dumps({
            "customer_id": "opaque-id",
            "label": 0,
            "label_name": "female",
            "sample_id": 0,
            "explanation": (
                "The client maintains regular transaction activity.\n"
                "Final: \\boxed{female}"
            ),
            "prompt_hash": "prompt",
            "client_stats_hash": "client",
            "summary_stats_hash": "summary",
            "generation_signature": "explanation",
            "min_behavioral_explanation_chars": 1,
            "error": None,
        }) + "\n",
        encoding="utf-8",
    )
    calls = 0

    async def always_empty(dialogues, model, llm_config, *, on_batch_complete):
        nonlocal calls
        calls += 1
        result = _result("[]")
        on_batch_complete([(0, result)])
        return [result]

    monkeypatch.setattr(
        "src.pipeline.claims_extractor.batched_query", always_empty
    )
    config = {
        "llm": {"default_model": "test"},
        "execution": {
            "until_complete": True,
            "content_primary_attempts": 1,
            "content_repair_attempts": 1,
        },
        "claims_generation": {"model": "test", "max_tokens": 128},
        "experiment": {"run_id": "test", "model_slug": "test"},
        "dataset": {
            "name": "gender",
            "label_names": {"0": "female", "1": "male"},
            "claim_forbidden_terms": [],
        },
        "pipeline": {"n_claims_samples": 1},
        "prompts": {
            "base_dir": ".",
            "claims_system": "prompts/common/claims_extraction/system_prompt.txt",
            "claims_user": "prompts/common/claims_extraction/user_prompt.txt",
        },
        "output": {
            "base_dir": str(tmp_path),
            "explanations": "explanations.jsonl",
            "claims": "claims.jsonl",
        },
    }

    run_claims_extraction(
        config, input_path=explanations, output_path=claims
    )

    assert calls == 2
    record = json.loads(claims.read_text(encoding="utf-8"))
    assert record["terminal_content_failure"] is True
    stats = json.loads(
        claims.with_suffix(".generation_stats.json").read_text(encoding="utf-8")
    )
    assert stats["content_validation"]["max_attempts_for_one_request"] == 2
    assert next(iter(stats["content_validation"]["attempts_by_request"])).startswith(
        "opaque-id:0:"
    )

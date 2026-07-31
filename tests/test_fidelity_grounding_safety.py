import pandas as pd
import pytest

from scripts.run_fidelity_analysis import load_cell, validate_selected_teacher
from scripts.run_grounding_judge import (
    SYSTEM_PROMPT,
    judgment_signature,
    parse_json_object,
    validate_judgment,
)
from scripts.summarize_grounding_judges import validate_judge_records
from src.experiments.artifacts import files_fingerprint


def test_fidelity_requires_exact_teacher_coverage(tmp_path):
    features = tmp_path / "features.parquet"
    teacher = tmp_path / "teacher.csv"
    pd.DataFrame({
        "customer_id": [1, 2], "label": [0, 1], "cot_a": [1, 0],
    }).to_parquet(features, index=False)
    pd.DataFrame({
        "customer_id": [1], "label": [0],
        "probability_0": [0.9], "probability_1": [0.1],
    }).to_csv(teacher, index=False)
    with pytest.raises(ValueError, match="coverage mismatch"):
        load_cell(features, teacher, split="test")


def test_fidelity_rejects_changed_selected_teacher(tmp_path):
    teacher = tmp_path / "teacher.csv"
    teacher.write_text("customer_id,label\n1,0\n", encoding="utf-8")
    paths = {"train": str(teacher), "val": str(teacher), "test": str(teacher)}
    selection = {"selected": {"file_hashes": files_fingerprint(paths.values())}}
    validate_selected_teacher(selection, paths)
    teacher.write_text("customer_id,label\n1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_selected_teacher(selection, paths)


def test_grounding_signature_binds_full_rendered_input():
    record = {
        "sample_id": "s1", "evidence_hash": "abc", "claim": "Has category A",
        "client_stats": "A: 3", "train_reference_summary": "reference",
        "field_semantics": "counts",
    }
    baseline = judgment_signature(record, "judge")
    for field, value in (
        ("claim", "Has category B"),
        ("client_stats", "A: 4"),
        ("train_reference_summary", "other reference"),
        ("field_semantics", "amounts"),
        ("evidence_hash", "def"),
    ):
        changed = dict(record)
        changed[field] = value
        assert judgment_signature(changed, "judge") != baseline


def _judge_frame():
    return pd.DataFrame([
        {"sample_id": "s1", "judge_name": "a", "evidence_hash": "h", "verdict": "supported"},
        {"sample_id": "s1", "judge_name": "b", "evidence_hash": "h", "verdict": "unsupported"},
    ])


def test_grounding_summary_rejects_duplicate_incomplete_and_mismatched_votes():
    validate_judge_records(_judge_frame(), {"a", "b"})

    duplicate = pd.concat([_judge_frame(), _judge_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_judge_records(duplicate, {"a", "b"})

    with pytest.raises(ValueError, match="Incomplete"):
        validate_judge_records(_judge_frame().iloc[[0]], {"a", "b"})

    mismatch = _judge_frame()
    mismatch.loc[1, "evidence_hash"] = "other"
    with pytest.raises(ValueError, match="Evidence hash mismatch"):
        validate_judge_records(mismatch, {"a", "b"})


def test_grounding_protocol_keeps_evidence_roles_and_verdicts_disjoint():
    prompt = " ".join(SYSTEM_PROMPT.split())
    assert "CLIENT TRANSACTION SUMMARY as the primary evidence" in prompt
    assert "only to verify explicit comparative claims" in prompt
    assert "FIELD SEMANTICS" in prompt
    assert "partially_supported" in prompt
    assert "not_verifiable" in prompt


def test_grounding_judgment_schema_is_strict():
    valid = {
        "verdict": "supported",
        "claim_type": "direct_observation",
        "confidence": 4,
        "evidence": "The supplied category is present.",
        "reason": "The claim is a qualitative paraphrase.",
    }
    assert validate_judgment(valid) is None
    assert validate_judgment({**valid, "confidence": 4.0}) == "confidence_not_integer"
    assert validate_judgment({**valid, "confidence": 6}) == "confidence_out_of_range"
    assert validate_judgment({**valid, "verdict": "maybe"}) == "invalid_verdict"
    assert validate_judgment({**valid, "claim_type": "guess"}) == "invalid_claim_type"


def test_grounding_parser_handles_provider_wrapped_singleton_object():
    direct, error = parse_json_object('{"verdict": "supported"}')
    assert error is None
    assert direct == {"verdict": "supported"}

    wrapped, error = parse_json_object('[{"verdict": "unsupported"}]')
    assert error is None
    assert wrapped == {"verdict": "unsupported"}

    parsed, error = parse_json_object(
        '[{"verdict": "supported"}, {"verdict": "unsupported"}]'
    )
    assert parsed is None
    assert error == "json_not_object:list"

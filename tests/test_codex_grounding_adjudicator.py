from __future__ import annotations

import json

import pytest

from scripts.run_codex_grounding_adjudicator import (
    BLINDED_FIELDS,
    blinded_task,
    load_resumed_chunk,
    opaque_task_id,
    validate_chunk,
)


def task(sample_id="sample-1"):
    return {
        "sample_id": sample_id,
        "evidence_hash": "evidence-1",
        "claim": "The client is active.",
        "client_stats": "count: 10",
        "train_reference_summary": "median count: 5",
        "field_semantics": "count means observed transactions",
        "run_name": "qwen",
        "judge_details": [{"judge_name": "secret", "verdict": "unsupported"}],
        "true_label": 1,
        "prediction": 0,
    }


def test_corrupt_codex_resume_is_rejected(tmp_path):
    response = tmp_path / "response.json"
    resume = tmp_path / "resume.json"
    response.write_text("{not-json", encoding="utf-8")
    resume.write_text(json.dumps({"chunk_signature": "expected"}), encoding="utf-8")

    assert load_resumed_chunk(
        expected=[task()],
        response_path=response,
        resume_path=resume,
        chunk_signature="expected",
        model="gpt-5.5",
        start=0,
    ) is None


def test_valid_codex_resume_is_reused(tmp_path):
    response = tmp_path / "response.json"
    resume = tmp_path / "resume.json"
    row = task()
    response.write_text(json.dumps({"adjudications": [{
        "task_id": opaque_task_id(row),
        "verdict": "supported",
        "confidence": 5,
        "reason": "The claim directly matches the supplied statistic.",
    }]}), encoding="utf-8")
    resume.write_text(json.dumps({"chunk_signature": "expected"}), encoding="utf-8")

    rows = load_resumed_chunk(
        expected=[row],
        response_path=response,
        resume_path=resume,
        chunk_signature="expected",
        model="gpt-5.5",
        start=0,
    )

    assert rows is not None
    assert rows[0]["evidence_hash"] == "evidence-1"
    assert rows[0]["model"] == "gpt-5.5"


def test_adjudication_task_uses_opaque_id_and_excludes_source_and_votes():
    row = task("qwen:age:test:client-7")
    visible = blinded_task(row)
    assert tuple(visible) == ("task_id", *BLINDED_FIELDS)
    rendered = json.dumps(visible)
    for forbidden in (
        "qwen", "gpt_oss", "judge_details", "unsupported",
        "true_label", "prediction", "qwen:age:test:client-7",
    ):
        assert forbidden not in rendered


def test_adjudication_rejects_boolean_confidence():
    row = task()
    payload = {"adjudications": [{
        "task_id": opaque_task_id(row),
        "verdict": "supported",
        "confidence": True,
        "reason": "Directly observed.",
    }]}
    with pytest.raises(ValueError, match="confidence"):
        validate_chunk([row], payload, model="gpt-5.5")

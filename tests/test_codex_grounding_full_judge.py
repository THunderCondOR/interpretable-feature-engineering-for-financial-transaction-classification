from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.run_codex_grounding_full_judge import (
    BLINDED_FIELDS,
    JUDGE_NAME,
    blinded_task,
    build_prompt,
    judgment_signature,
    load_resumed_chunk,
    opaque_task_id,
    repair_incomplete_chunk,
    validate_chunk,
    validate_complete,
    validate_partial_chunk,
)
from scripts.summarize_grounding_three_judges import aggregate_three_judges
from scripts.summarize_grounding_three_judges import build_summary, fleiss_kappa


def sample(sample_id: str = "s1") -> dict:
    return {
        "sample_id": sample_id,
        "dataset": "datafusion_default_2023",
        "run_name": "secret-source-model",
        "customer_id": "client-1",
        "true_label": 1,
        "prediction": 0,
        "judge_verdicts": {"j1": "supported"},
        "claim": "The client has frequent transactions.",
        "client_stats": "transaction count: 10",
        "train_reference_summary": "median count: 5",
        "field_semantics": "count is the number of observed transactions",
        "evidence_hash": "hash-1",
    }


def response(row: dict | None = None, verdict: str = "supported") -> dict:
    row = row or sample()
    return {"judgments": [{
        "task_id": opaque_task_id(row),
        "verdict": verdict,
        "claim_type": "train_relative_comparison",
        "confidence": 4,
        "evidence": "10 transactions versus a train median of 5.",
        "reason": "The supplied values support the comparison.",
    }]}


def test_full_codex_task_and_prompt_are_blinded(tmp_path):
    row = sample("qwen:age:test:customer-42")
    visible = blinded_task(row)
    assert tuple(visible) == ("task_id", *BLINDED_FIELDS)
    dumped = json.dumps(visible)
    for secret in (
        "true_label", "prediction", "secret-source-model", "judge_verdicts",
        "qwen", "gpt_oss", "qwen:age:test:customer-42",
    ):
        assert secret not in dumped
    prompt = build_prompt(tmp_path / "tasks.json")
    assert "only the JSON tasks" in prompt
    assert "judge verdict" not in prompt.lower()
    assert "qwen" not in prompt.lower()
    assert "gpt_oss" not in prompt.lower()


def test_full_codex_chunk_requires_exact_ids_and_preserves_signatures():
    samples = [sample("s1"), sample("s2")]
    payload = {
        "judgments": response(samples[0])["judgments"]
        + response(samples[1])["judgments"]
    }
    rows = validate_chunk(samples, payload)
    validate_complete(samples, rows)
    assert {row["sample_id"] for row in rows} == {"s1", "s2"}
    assert rows[0]["evidence_hash"] == "hash-1"
    assert rows[0]["judgment_signature"] == judgment_signature(samples[0])
    with pytest.raises(ValueError, match="incomplete"):
        validate_chunk(samples, response(samples[0]))
    partial = validate_partial_chunk(samples, response(samples[0]))
    assert [row["sample_id"] for row in partial] == ["s1"]


def test_repair_preserves_valid_partial_and_requests_only_missing(tmp_path, monkeypatch):
    samples = [sample("s1"), sample("s2"), sample("s3")]
    response_path = tmp_path / "response.json"
    tasks_path = tmp_path / "tasks.json"
    schema_path = tmp_path / "schema.json"
    response_path.write_text(json.dumps(response(samples[0])), encoding="utf-8")
    seen = []

    def fake_run(*, expected, tasks_path, response_path, schema_path):
        seen.append([row["sample_id"] for row in expected])
        return validate_partial_chunk(
            expected,
            {"judgments": [response(row)["judgments"][0] for row in expected]},
        )

    monkeypatch.setattr(
        "scripts.run_codex_grounding_full_judge.run_codex_chunk", fake_run
    )
    rows = repair_incomplete_chunk(
        expected=samples, tasks_path=tasks_path, response_path=response_path,
        schema_path=schema_path, max_attempts=2,
    )
    assert seen == [["s2", "s3"]]
    assert [row["sample_id"] for row in rows] == ["s1", "s2", "s3"]
    canonical = json.loads(response_path.read_text(encoding="utf-8"))
    assert len(canonical["judgments"]) == 3


def test_corrupt_or_stale_full_codex_resume_is_rejected(tmp_path):
    response_path = tmp_path / "response.json"
    resume_path = tmp_path / "resume.json"
    response_path.write_text("{bad-json", encoding="utf-8")
    resume_path.write_text(json.dumps({"chunk_signature": "sig"}), encoding="utf-8")
    assert load_resumed_chunk(
        expected=[sample()], response_path=response_path,
        resume_path=resume_path, chunk_signature="sig", start=0,
    ) is None
    response_path.write_text(json.dumps(response(sample())), encoding="utf-8")
    resume_path.write_text(json.dumps({"chunk_signature": "stale"}), encoding="utf-8")
    assert load_resumed_chunk(
        expected=[sample()], response_path=response_path,
        resume_path=resume_path, chunk_signature="sig", start=0,
    ) is None


def test_three_judge_aggregation_uses_majority_and_keeps_three_way_tie():
    samples = [sample("majority"), sample("tie")]
    judgments = []
    verdicts = {
        "majority": ["supported", "partially_supported", "supported"],
        "tie": ["supported", "partially_supported", "unsupported"],
    }
    for sample_id, votes in verdicts.items():
        for judge, verdict in zip(("judge_a", "judge_b", JUDGE_NAME), votes):
            judgments.append({
                "sample_id": sample_id,
                "judge_name": judge,
                "verdict": verdict,
                "claim_type": "direct_observation",
            })
    items, metrics = aggregate_three_judges(samples, pd.DataFrame(judgments))
    by_id = items.set_index("sample_id")
    assert by_id.loc["majority", "final_verdict"] == "supported"
    assert bool(by_id.loc["majority", "has_majority"])
    assert by_id.loc["tie", "final_verdict"] == "disagreement"
    assert not bool(by_id.loc["tie", "has_majority"])
    assert metrics["three_way_no_majority_share"] == 0.5


def test_fleiss_kappa_and_cell_shares_for_complete_agreement():
    samples = [sample("all-supported"), sample("all-unsupported")]
    judgments = []
    for sample_id, verdict in (
        ("all-supported", "supported"),
        ("all-unsupported", "unsupported"),
    ):
        for judge in ("judge_a", "judge_b", JUDGE_NAME):
            judgments.append({
                "sample_id": sample_id,
                "judge_name": judge,
                "verdict": verdict,
                "claim_type": "direct_observation",
            })
    frame = pd.DataFrame(judgments)
    assert fleiss_kappa(
        frame, value_column="verdict",
        categories=("supported", "partially_supported", "unsupported", "not_verifiable"),
    ) == pytest.approx(1.0)
    items, metrics = aggregate_three_judges(samples, frame)
    assert metrics["fleiss_kappa_verdict"] == pytest.approx(1.0)
    cell = metrics["cells"]["secret-source-model/datafusion_default_2023"]
    assert cell["verdicts"]["supported"]["share"] == 0.5
    assert cell["verdicts"]["unsupported"]["share"] == 0.5
    summary = build_summary(items)
    assert summary.loc[0, "verdict_supported_share"] == 0.5
    assert summary.loc[0, "claim_type_direct_observation_share"] == 1.0

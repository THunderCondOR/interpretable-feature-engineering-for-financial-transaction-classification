import json
from types import SimpleNamespace

from src.experiments.artifacts import fingerprint, prompt_signature
from src.pipeline.explanation_gen import (
    _is_successful,
    build_output_record,
    run_explanation_generation,
    summarize_records,
    write_records,
)


LABELS = {"0": "female", "1": "male"}
META = {"customer_id": 1, "label": 0, "label_name": "female", "sample_id": 0}
VALID_RESPONSE = (
    "The client regularly transacts across several categories. "
    "Activity and spending composition remain stable over the observed period.\n"
    "Final: \\boxed{female}"
)


def api_result(
    content: str | None,
    *,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    reasoning: str | None = None,
) -> dict:
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        model_extra={"reasoning": reasoning} if reasoning is not None else {},
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    response = SimpleNamespace(choices=[choice], usage=usage)
    return {"response": response, "execution_time": 1.5, "error": None}


def test_complete_boxed_response_is_successful() -> None:
    record = build_output_record(
        META,
        api_result(VALID_RESPONSE),
        LABELS,
    )

    assert _is_successful(record)
    assert record["predicted"] == 0
    assert record["finish_reason"] == "stop"
    assert record["completion_tokens"] == 20


def test_empty_content_with_reasoning_is_not_successful() -> None:
    record = build_output_record(
        META,
        api_result("", reasoning_content="Internal reasoning"),
        LABELS,
    )

    assert not _is_successful(record)
    assert record["error"] == "empty response content"
    assert record["error_type"] == "EmptyResponse"
    assert record["reasoning"] == "Internal reasoning"
    assert record["explanation"] == ""
    assert record["reasoning_chars"] > 0


def test_provider_reasoning_is_used_when_visible_content_has_only_final() -> None:
    provider_reasoning = (
        "Transaction activity is sparse and declines late in the observation "
        "period, which is more consistent with the first class."
    )
    record = build_output_record(
        META,
        api_result(
            "Final: \\boxed{female}",
            reasoning=provider_reasoning,
        ),
        LABELS,
    )

    assert _is_successful(record)
    assert record["predicted"] == 0
    assert record["explanation"] == (
        f"{provider_reasoning}\n\nFinal: \\boxed{{female}}"
    )
    assert record["response_content"] == "Final: \\boxed{female}"
    assert record["reasoning"] == provider_reasoning
    assert record["reasoning_field"] == "reasoning"
    assert record["explanation_source"] == "provider_reasoning_fallback"


def test_provider_reasoning_is_used_when_visible_content_repeats_label() -> None:
    provider_reasoning = (
        "The client has regular transaction activity across several observed "
        "categories and maintains a stable cash-flow pattern."
    )
    record = build_output_record(
        META,
        api_result(
            "female\n\nFinal: \\boxed{female}",
            reasoning=provider_reasoning,
        ),
        LABELS,
    )

    assert _is_successful(record)
    assert record["explanation"].startswith(provider_reasoning)
    assert record["explanation_source"] == "provider_reasoning_fallback"


def test_visible_behavioral_explanation_is_preferred_over_provider_reasoning() -> None:
    record = build_output_record(
        META,
        api_result(VALID_RESPONSE, reasoning="Private scratch analysis."),
        LABELS,
    )

    assert _is_successful(record)
    assert record["explanation"] == VALID_RESPONSE
    assert record["reasoning"] == "Private scratch analysis."
    assert record["explanation_source"] == "content"


def test_truncated_response_is_not_successful() -> None:
    record = build_output_record(
        META,
        api_result("Incomplete rationale", finish_reason="length"),
        LABELS,
    )

    assert not _is_successful(record)
    assert record["error"] == "incomplete response: finish_reason=length"
    assert record["error_type"] == "IncompleteResponse"


def test_response_without_boxed_answer_is_not_successful() -> None:
    record = build_output_record(META, api_result("Probably female."), LABELS)

    assert not _is_successful(record)
    assert record["error"] == "missing or unrecognized boxed final answer"
    assert record["error_type"] == "MissingFinalAnswer"


def test_short_behavioral_rationale_is_valid_but_final_only_is_not() -> None:
    short = build_output_record(
        META,
        api_result("Low transaction activity.\nFinal: \\boxed{female}"),
        LABELS,
    )
    final_only = build_output_record(
        META,
        api_result("Final: \\boxed{female}"),
        LABELS,
    )
    assert _is_successful(short)
    assert not _is_successful(final_only)
    assert final_only["error_type"] == "NoBehavioralExplanation"


def test_write_records_is_atomic_and_supports_partial_checkpoint(tmp_path) -> None:
    record = build_output_record(
        META,
        api_result(VALID_RESPONSE),
        LABELS,
    )
    path = tmp_path / "explanations.jsonl"

    assert write_records(path, [META], {(1, 0): record}) == 0
    assert path.exists()
    assert not path.with_suffix(".jsonl.tmp").exists()
    assert '"predicted": 0' in path.read_text(encoding="utf-8")


def test_summarize_records_counts_error_types() -> None:
    good = build_output_record(
        META,
        api_result(VALID_RESPONSE),
        LABELS,
    )
    empty = build_output_record(META, api_result(""), LABELS)
    truncated = build_output_record(
        META,
        api_result("Incomplete rationale", finish_reason="length"),
        LABELS,
    )

    assert summarize_records([good, empty, truncated]) == {
        "total": 3,
        "successful": 1,
        "failed": 2,
        "error_types": {
            "EmptyResponse": 1,
            "IncompleteResponse": 1,
        },
    }


def test_resume_is_default_and_reuses_existing_record(tmp_path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "explanations.jsonl"
    prompt_path.write_text(
        json.dumps(
            {
                "customer_id": 1,
                "label": 0,
                "label_name": "female",
                "system_prompt": "system",
                "user_prompt": "user",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    signature = prompt_signature(
        system_prompt="system",
        user_prompt="user",
        model="test",
        decoding={
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 2048,
            "seed": None,
            "extra_body": None,
        },
        sample_id=0,
    )
    existing_meta = {
        **META,
        "generation_signature": signature,
        "prompt_hash": fingerprint(
            {"system_prompt": "system", "user_prompt": "user"}
        ),
    }
    existing = build_output_record(
        existing_meta,
        api_result(VALID_RESPONSE),
        LABELS,
    )
    write_records(output_path, [META], {(1, 0): existing})

    async def unexpected_api_call(*args, **kwargs):
        raise AssertionError("API must not be called when the existing record is valid")

    monkeypatch.setattr(
        "src.pipeline.explanation_gen.batched_query",
        unexpected_api_call,
    )
    config = {
        "llm": {"default_model": "test"},
        "dataset": {"label_names": LABELS},
        "pipeline": {"n_explanation_samples": 1},
        "output": {
            "base_dir": str(tmp_path),
            "prompts": "prompts.jsonl",
            "explanations": "explanations.jsonl",
        },
    }

    run_explanation_generation(
        config,
        input_path=prompt_path,
        output_path=output_path,
    )

    stats = json.loads(
        output_path.with_suffix(".generation_stats.json").read_text(encoding="utf-8")
    )
    assert stats["resume"] is True
    assert stats["reused_successful"] == 1
    assert stats["planned_new_requests"] == 0


def test_batch_error_summary_is_saved_by_type(tmp_path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "explanations.jsonl"
    prompts = [
        {
            "customer_id": customer_id,
            "label": 0,
            "label_name": "female",
            "system_prompt": "system",
            "user_prompt": "user",
        }
        for customer_id in (1, 2)
    ]
    prompt_path.write_text(
        "".join(json.dumps(prompt, ensure_ascii=False) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    results = [
        api_result(VALID_RESPONSE),
        api_result(""),
    ]

    async def fake_batched_query(dialogues, model, llm_config, *, on_batch_complete):
        indexed = list(enumerate(results))
        on_batch_complete(indexed)
        return results

    monkeypatch.setattr(
        "src.pipeline.explanation_gen.batched_query",
        fake_batched_query,
    )
    config = {
        "llm": {"default_model": "test"},
        "dataset": {"label_names": LABELS},
        "pipeline": {"n_explanation_samples": 1},
        "output": {
            "base_dir": str(tmp_path),
            "prompts": "prompts.jsonl",
            "explanations": "explanations.jsonl",
        },
    }

    run_explanation_generation(
        config,
        input_path=prompt_path,
        output_path=output_path,
    )

    stats = json.loads(
        output_path.with_suffix(".generation_stats.json").read_text(encoding="utf-8")
    )
    assert stats["new_requests"] == {
        "total": 2,
        "successful": 1,
        "failed": 1,
        "error_types": {"EmptyResponse": 1},
    }
    assert stats["last_batch"] == stats["new_requests"]


def test_until_complete_commits_good_records_and_defers_only_bad_content(
    tmp_path, monkeypatch
) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "explanations.jsonl"
    prompts = [
        {
            "customer_id": customer_id,
            "label": 0,
            "label_name": "female",
            "system_prompt": "system",
            "user_prompt": f"user {customer_id}",
        }
        for customer_id in (1, 2)
    ]
    prompt_path.write_text(
        "".join(json.dumps(prompt) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    call_sizes = []

    async def fake_batched_query(
        dialogues, model, llm_config, *, on_batch_complete
    ):
        call_sizes.append(len(dialogues))
        results = (
            [api_result(VALID_RESPONSE), api_result("")]
            if len(call_sizes) == 1
            else [api_result(VALID_RESPONSE)]
        )
        on_batch_complete(list(enumerate(results)))
        return results

    monkeypatch.setattr(
        "src.pipeline.explanation_gen.batched_query",
        fake_batched_query,
    )
    config = {
        "llm": {"default_model": "test"},
        "execution": {"until_complete": True},
        "experiment": {"run_id": "test", "model_slug": "test"},
        "dataset": {"name": "gender", "label_names": LABELS},
        "pipeline": {"n_explanation_samples": 1},
        "output": {
            "base_dir": str(tmp_path),
            "prompts": "prompts.jsonl",
            "explanations": "explanations.jsonl",
        },
    }

    run_explanation_generation(
        config,
        input_path=prompt_path,
        output_path=output_path,
    )

    assert call_sizes == [2, 1]
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert all(_is_successful(record) for record in records)
    stats = json.loads(
        output_path.with_suffix(".generation_stats.json").read_text(
            encoding="utf-8"
        )
    )
    assert stats["content_validation"] == {
        "total_rejections": 1,
        "unique_rejected_requests": 1,
        "error_types": {"EmptyResponse": 1},
        "max_attempts_for_one_request": 1,
        "attempts_by_request": {"2:0": 1},
        "last_reasons_by_request": {"2:0": "empty response content"},
    }
    events = [
        json.loads(line)
        for line in (
            tmp_path / ".scheduler" / "explanations.events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    deferred = [
        event for event in events if event["event"] == "content_records_deferred"
    ]
    assert deferred[0]["accepted"] == 1
    assert deferred[0]["deferred"] == 1
    assert deferred[0]["errors"][0]["request_key"] == "2:0"


def test_content_failure_has_finite_attempt_limit(tmp_path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "explanations.jsonl"
    prompt_path.write_text(
        json.dumps({
            "customer_id": 1,
            "label": 0,
            "label_name": "female",
            "system_prompt": "system",
            "user_prompt": "user",
        }) + "\n",
        encoding="utf-8",
    )
    calls = 0

    async def always_empty(dialogues, model, llm_config, *, on_batch_complete):
        nonlocal calls
        calls += 1
        result = api_result("")
        on_batch_complete([(0, result)])
        return [result]

    monkeypatch.setattr(
        "src.pipeline.explanation_gen.batched_query", always_empty
    )
    config = {
        "llm": {"default_model": "test"},
        "execution": {
            "until_complete": True,
            "content_primary_attempts": 1,
            "content_repair_attempts": 1,
        },
        "experiment": {"run_id": "test", "model_slug": "test"},
        "dataset": {"name": "gender", "label_names": LABELS},
        "pipeline": {"n_explanation_samples": 1},
        "output": {
            "base_dir": str(tmp_path),
            "prompts": "prompts.jsonl",
            "explanations": "explanations.jsonl",
        },
    }

    run_explanation_generation(
        config, input_path=prompt_path, output_path=output_path
    )

    assert calls == 2
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["terminal_content_failure"] is True
    assert record["content_attempts"] == 2

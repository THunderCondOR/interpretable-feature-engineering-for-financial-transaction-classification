from types import SimpleNamespace

from scripts.stress_test_llm import summarize_results


def test_summarize_results_counts_transport_and_response_metrics() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="ok"),
            )
        ],
        usage=SimpleNamespace(completion_tokens=10),
    )
    results = [
        {
            "response": response,
            "execution_time": 2.0,
            "error": None,
            "error_type": None,
        },
        {
            "response": None,
            "execution_time": 0.0,
            "error": "rate limited",
            "error_type": "RateLimitError",
        },
    ]

    summary = summarize_results(
        results,
        concurrency=2,
        request_count=2,
        wall_seconds=4.0,
    )

    assert summary["requests_per_second"] == 0.5
    assert summary["transport_successful"] == 1
    assert summary["transport_failed"] == 1
    assert summary["transport_error_types"] == {"RateLimitError": 1}
    assert summary["finish_reasons"] == {"stop": 1}
    assert summary["completion_tokens"]["mean"] == 10.0

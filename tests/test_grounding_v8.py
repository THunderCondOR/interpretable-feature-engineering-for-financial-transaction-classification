import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.build_grounding_annotation import select_blinded
from scripts.run_grounding_judge import make_dialogue
from scripts.run_grounding_suite import DEFAULT_JUDGES, estimate_cost
from src.utils.async_api import _make_client


ROOT = Path(__file__).parents[1]


def _sample(sample_id="s", dataset="rosbank", run_name="qwen"):
    return {
        "sample_id": sample_id, "dataset": dataset, "run_name": run_name,
        "customer_id": sample_id, "client_stats": "client-only-secret",
        "train_reference_summary": "shared-reference", "field_semantics": "amount",
        "claim": "The client is active.", "evidence_hash": "hash",
    }


def test_grounding_dialogue_is_blinded_and_cacheable():
    row = {**_sample(), "true_label": 1, "prediction": 0}
    text = make_dialogue(row)[1]["content"]
    assert text.index("shared-reference") < text.index("client-only-secret")
    assert "qwen" not in text
    assert "true_label" not in text
    assert "prediction" not in text


def test_openai_client_uses_proxy_only_when_explicitly_enabled():
    base = {
        "api_base_url": "https://example.test/v1",
        "api_key": "not-a-secret",
        "max_concurrent": 2,
    }
    with patch("src.utils.async_api.openai.AsyncOpenAI"), patch(
        "src.utils.async_api.httpx.AsyncClient"
    ) as client:
        _make_client(base)
        assert client.call_args.kwargs["trust_env"] is False
    with patch("src.utils.async_api.openai.AsyncOpenAI"), patch(
        "src.utils.async_api.httpx.AsyncClient"
    ) as client:
        _make_client({**base, "use_env_proxy": True})
        assert client.call_args.kwargs["trust_env"] is True


def test_human_annotation_is_balanced_and_hides_source_model():
    rows = []
    for dataset in ("rosbank", "berka"):
        for model in ("qwen", "gpt_oss"):
            rows.extend(_sample(f"{dataset}-{model}-{i}", dataset, model) for i in range(20))
    visible, key = select_blinded(rows, per_dataset=10, seed=17)
    assert len(visible) == 20
    assert all("run_name" not in row and "sample_id" not in row for row in visible)
    assert {row["run_name"] for row in key} == {"qwen", "gpt_oss"}
    assert sum(row["phase"] == "calibration" for row in visible) == 6


def test_two_judge_cost_estimate_is_conservative(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    result = estimate_cost(path, DEFAULT_JUDGES, max_tokens=192)
    assert result["estimated_input_tokens_per_judge"] > 0
    assert result["maximum_output_tokens_per_judge"] == 192
    assert result["estimated_total_usd"] == sum(
        row["estimated_usd"] for row in result["models"].values()
    )


def test_codex_adjudication_is_merged_without_overwriting_consensus(tmp_path):
    inputs = []
    for judge, verdicts in (("j1", ["supported", "supported"]), ("j2", ["supported", "unsupported"])):
        path = tmp_path / f"{judge}.jsonl"
        records = []
        for index, verdict in enumerate(verdicts):
            records.append({
                **_sample(f"s{index}"), "judge_name": judge, "verdict": verdict,
                "confidence": 4, "evidence": "e", "reason": "r",
            })
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        inputs.append(path)
    prefix = tmp_path / "grounding"
    subprocess.run([
        sys.executable, "scripts/summarize_grounding_judges.py",
        "--inputs", *map(str, inputs), "--expected-judges", "j1", "j2",
        "--output-prefix", str(prefix), "--execute",
    ], cwd=ROOT, check=True)
    tasks = json.loads(prefix.with_suffix(".disagreements.json").read_text())
    assert [row["sample_id"] for row in tasks] == ["s1"]
    adjudication = tmp_path / "codex.json"
    adjudication.write_text(json.dumps([{
        "sample_id": "s1", "verdict": "partially_supported",
        "confidence": 4, "reason": "partly inferred",
    }]), encoding="utf-8")
    subprocess.run([
        sys.executable, "scripts/summarize_grounding_judges.py",
        "--inputs", *map(str, inputs), "--expected-judges", "j1", "j2",
        "--output-prefix", str(prefix), "--adjudication", str(adjudication), "--execute",
    ], cwd=ROOT, check=True)
    rows = prefix.with_suffix(".items.csv").read_text(encoding="utf-8")
    assert "initial_consensus" in rows
    assert "codex_adjudication" in rows
    assert "partially_supported" in rows

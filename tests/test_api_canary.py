import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.api_canary import run_model_canary


ROOT = Path(__file__).parents[1]


class _FakeCompletions:
    def __init__(self):
        self.calls = []
        self.responses = [
            (
                "The client shows sustained transaction activity across several "
                "categories. The operation mix remains diverse throughout the "
                "observed period.\nFinal: \\boxed{retained_client}"
            ),
            json.dumps(
                [
                    "The client shows sustained transaction activity.",
                    "The client transacts across diverse categories.",
                ]
            ),
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        message = SimpleNamespace(
            content=content,
            reasoning_content=None,
            model_extra={},
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )


def _config() -> dict:
    return {
        "experiment": {
            "model_slug": "qwen",
            "variant": "guided_zero_shot_v4",
            "label_semantics": "standard",
        },
        "dataset": {
            "name": "rosbank",
            "label_names": {"0": "retained_client", "1": "churned_client"},
            "claim_forbidden_terms": ["retained", "churned", "churn"],
        },
        "generation": {
            "model": "Qwen/Test",
            "temperature": 0.8,
            "top_p": 0.9,
            "seed": 17,
        },
        "claims_generation": {
            "model": "Qwen/Test",
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 17,
        },
        "llm": {
            "max_tokens": 1024,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "pipeline": {
            "claims_max_tokens": 256,
            "min_behavioral_explanation_chars": 80,
        },
        "prompts": {
            "base_dir": ".",
            "claims_system": "prompts/common/claims_extraction/system_prompt.txt",
            "claims_user": "prompts/common/claims_extraction/user_prompt.txt",
        },
    }


def test_canary_validates_real_explanation_to_claims_contract():
    completions = _FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    prompt = {
        "customer_id": 7,
        "label": 0,
        "label_name": "retained_client",
        "system_prompt": "Use only supplied evidence.",
        "user_prompt": "A transaction profile.",
    }
    result = run_model_canary(
        config=_config(),
        prompt_record=prompt,
        client=client,
    )
    assert result["prediction_parsed"] is True
    assert result["claims_parsed"] == 2
    claims_prompt = completions.calls[1]["messages"][1]["content"]
    assert "Final:" not in claims_prompt
    assert "- retained_client" in claims_prompt
    assert "retained_client" not in claims_prompt.split("RATIONALE", 1)[1]
    assert completions.calls[1]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_canary_dry_run_does_not_read_configs_or_call_api(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/api_canary.py",
            "--model-config",
            str(tmp_path / "missing.yaml"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["mode"] == "dry-run"
    assert plan["writes_experiment_results"] is False

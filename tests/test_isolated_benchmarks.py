import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_isolated_prompt_pilot import (
    promote_selected_qwen_pilot,
    reuse_compatible_pilot_generation,
)
from scripts.audit_isolated_prompts import _audit_ids
from src.benchmarks import cofinfad, datafusion_default_2023 as df23
from src.benchmarks.common import write_ids
from src.data.profiles import format_client_profile
from src.experiments.artifacts import file_sha256, prompt_signature
from src.experiments.config_builder import load_yaml
from src.pipeline.prompt_builder import validate_prompt_contract


def test_write_ids_returns_path_and_deduplicates(tmp_path):
    path = tmp_path / "ids.json"
    assert write_ids(path, [2, "1", 2]) == path
    assert json.loads(path.read_text()) == ["1", "2"]


def test_datafusion_fixed_split_and_pilot_cover_rare_class():
    labels = pd.DataFrame({
        "customer_id": [str(value) for value in range(df23.EXPECTED_CLIENTS)],
        "label": [1] * df23.EXPECTED_POSITIVES
        + [0] * (df23.EXPECTED_CLIENTS - df23.EXPECTED_POSITIVES),
    })
    roles = df23._split(labels)
    assert {role: len(ids) for role, ids in roles.items()} == {
        "train": 4248, "val": 1416, "test": 1416,
    }
    assert not (set(roles["train"]) & set(roles["val"]))
    events = pd.DataFrame({
        "customer_id": labels["customer_id"],
        "amount": np.linspace(-10, 10, len(labels)),
    })
    pilot = df23._pilot_ids(events, labels, roles["val"])
    val_positive = set(labels.loc[
        labels["customer_id"].isin(roles["val"]) & labels["label"].eq(1),
        "customer_id",
    ])
    assert len(pilot) == 400
    assert val_positive <= set(pilot)


def test_prompt_audit_preserves_integer_entity_ids():
    frame = pd.DataFrame({
        "customer_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "label": [0, 0, 0, 0, 1, 1, 1, 1],
        "amount": [1.5, 2.5, 2.0, 3.0, 10.0, 20.0, 5.0, 7.0],
    })
    selected = _audit_ids(frame)
    assert selected
    assert all(isinstance(value, int) for value in selected)


def test_new_prompt_contracts_are_english_and_supported():
    for path in (
        Path("configs/datafusion_default_2023/base.yaml"),
        Path("configs/cofinfad/base.yaml"),
    ):
        validate_prompt_contract(load_yaml(path))


def test_datafusion_profile_separates_signs_and_currencies():
    config = load_yaml("configs/datafusion_default_2023/base.yaml")
    frame = pd.DataFrame({
        "customer_id": ["a"] * 4,
        "label": [0] * 4,
        "tr_datetime": pd.to_datetime(["2023-01-01"] * 4),
        "amount": [10.0, -5.0, 20.0, -2.0],
        "mcc_code_desc": ["Groceries", "Groceries", "Travel", "Travel"],
        "currency_name": ["RUR", "RUR", "USD", "USD"],
    })
    rendered = format_client_profile(frame, config)
    assert "RUR signed-direction statistics" in rendered
    assert "USD signed-direction statistics" in rendered
    assert "total income" not in rendered.lower()
    assert "total expenses" not in rendered.lower()
    assert "positive and negative directions are factual" in rendered


def test_cofinfad_profile_excludes_demographics_and_teacher_target():
    config = load_yaml("configs/cofinfad/base.yaml")
    frame = pd.DataFrame({
        "customer_id": ["a", "a"], "label": [2, 2],
        "tr_datetime": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "amount": [100.0, 200.0],
        "mcc_code_desc": ["Transfer", "Payment"],
        "transaction_type": ["Transfer", "Payment"],
        "active_products": [2, 2], "savings_account": [True, True],
        "age": [91, 91], "gender": ["secret", "secret"],
        "churn_probability": [0.91, 0.91],
    })
    rendered = format_client_profile(frame, config).lower()
    assert "active products: 2" in rendered
    assert "age:" not in rendered
    assert "gender:" not in rendered
    assert "churn probability" not in rendered
    assert "0.91" not in rendered


def test_pilot_promotion_is_idempotent_and_content_addressed(tmp_path):
    source_root = tmp_path / "pilot"
    full_root = tmp_path / "full"
    source_root.mkdir()
    source = source_root / "explanations_val.jsonl"
    rows = [{"customer_id": 7, "explanation": "Observed behavior. Final: x", "predicted": 0}]
    source.write_text(json.dumps(rows[0]) + "\n")
    pilot_config = tmp_path / "pilot.yaml"
    pilot_config.write_text(
        "output:\n  paths_by_split:\n    val:\n      explanations: " + str(source) + "\n"
    )
    selected_config = {
        "output": {"paths_by_split": {"val": {"explanations": str(full_root / "explanations_val.jsonl")}}}
    }
    payload = {
        "dataset": "datafusion_default_2023", "run_id": "test",
        "pilot_ids_hash": "ids", "pilot_configs": {"guided_zero_shot_v5": str(pilot_config)},
    }
    first = promote_selected_qwen_pilot(
        selected_variant="guided_zero_shot_v5", payload=payload,
        selected_config=selected_config, rows=rows, cell=tmp_path,
    )
    second = promote_selected_qwen_pilot(
        selected_variant="guided_zero_shot_v5", payload=payload,
        selected_config=selected_config, rows=rows, cell=tmp_path,
    )
    assert first["destination_sha256"] == second["destination_sha256"]
    assert first["source_sha256"] == file_sha256(source)


def test_pilot_generation_reuse_recomputes_metrics_after_evaluator_change(
    tmp_path,
):
    prompt = {
        "customer_id": 7,
        "system_prompt": "system",
        "user_prompt": "user",
        "prompt_hash": "prompt-hash",
    }
    decoding = {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 128,
        "seed": 17,
        "extra_body": None,
    }
    explanation = {
        "customer_id": 7,
        "label": 0,
        "predicted": 0,
        "explanation": (
            "The client has regular observed transaction activity.\n"
            "Final: \\boxed{no_default}"
        ),
        "prompt_hash": "prompt-hash",
        "generation_signature": prompt_signature(
            system_prompt="system",
            user_prompt="user",
            model="test-model",
            decoding=decoding,
            sample_id=0,
        ),
    }
    (tmp_path / "prompts_val.jsonl").write_text(
        json.dumps(prompt) + "\n", encoding="utf-8"
    )
    (tmp_path / "explanations_val.jsonl").write_text(
        json.dumps(explanation) + "\n", encoding="utf-8"
    )
    config = {
        "llm": {"default_model": "test-model", "max_tokens": 128},
        "generation": {"temperature": 0.0, "top_p": 0.9, "seed": 17},
        "output": {
            "base_dir": str(tmp_path),
            "explanations": "explanations.jsonl",
        },
    }

    assert reuse_compatible_pilot_generation(config, {7})
    metrics = json.loads((tmp_path / "llm_metrics_val.json").read_text())
    assert metrics["n_scored"] == 1
    assert metrics["coverage"] == 1.0


def test_cofinfad_risk_boundaries_are_ordered():
    scores = pd.Series([0.0, 0.2, 0.5, 0.8, 1.0])
    assert cofinfad._risk_labels(scores, [0.25, 0.5, 0.75]).tolist() == [0, 0, 2, 3, 3]

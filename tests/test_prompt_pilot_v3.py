import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.prompt_pilot import (
    AGE_OPAQUE,
    AGE_ORDERED,
    FS1,
    FS2,
    ZERO_SHOT,
    paired_balanced_accuracy_delta,
    rationale_diagnostics,
    select_age_label_semantics,
    select_prompt_variant,
)
from scripts.run_prompt_pilot import PILOT_VARIANTS, stratified_pilot_ids


ROOT = Path(__file__).parents[1]


def _metrics(zero=0.70, fs1=0.70, fs2=0.70):
    return {
        ZERO_SHOT: {"balanced_accuracy": zero},
        FS1: {"balanced_accuracy": fs1},
        FS2: {"balanced_accuracy": fs2},
    }


def _deltas(fs1_low=0.01, fs2_low=0.01):
    return {
        FS1: {"ci_low": fs1_low},
        FS2: {"ci_low": fs2_low},
    }


def test_few_shot_gain_must_be_strictly_more_than_two_points():
    exact = select_prompt_variant(
        _metrics(fs1=0.72, fs2=0.70), _deltas(fs1_low=0.001)
    )
    above = select_prompt_variant(
        _metrics(fs1=0.720001, fs2=0.70), _deltas(fs1_low=0.001)
    )
    assert exact["selected_variant"] == ZERO_SHOT
    assert above["selected_variant"] == FS1


def test_few_shot_must_have_positive_paired_ci_lower_bound():
    decision = select_prompt_variant(
        _metrics(fs1=0.75, fs2=0.70), _deltas(fs1_low=0.0)
    )
    assert decision["selected_variant"] == ZERO_SHOT


def test_fs1_wins_when_qualified_few_shots_are_within_half_point():
    decision = select_prompt_variant(
        _metrics(fs1=0.751, fs2=0.755, zero=0.70),
        _deltas(fs1_low=0.01, fs2_low=0.01),
    )
    assert decision["selected_variant"] == FS1


def test_paired_bootstrap_requires_identical_client_ids_and_detects_gain():
    zero = [
        {"customer_id": i, "label": i % 2, "predicted": 0}
        for i in range(20)
    ]
    candidate = [
        {"customer_id": i, "label": i % 2, "predicted": i % 2}
        for i in range(20)
    ]
    result = paired_balanced_accuracy_delta(
        zero, candidate, samples=100, seed=9
    )
    assert np.isclose(result["delta"], 0.5)
    assert result["ci_low"] > 0


def test_rationale_diversity_and_social_audit_are_reported():
    explanations = [
        {
            "customer_id": 1,
            "predicted": 0,
            "error": None,
            "explanation": (
                "Frequent transactions in the books category. "
                "The client works as a doctor. Final: label"
            ),
        },
        {
            "customer_id": 2,
            "predicted": 1,
            "error": None,
            "explanation": "Infrequent transactions and low value. Final: label",
        },
    ]
    prompts = [
        {"customer_id": 1, "client_stats": "  - books: 3"},
        {"customer_id": 2, "client_stats": "  - travel: 1"},
    ]
    diagnostics = rationale_diagnostics(explanations, prompts)
    assert diagnostics["parse_success"] == 1.0
    assert diagnostics["lexical_type_token_ratio"] > 0
    assert diagnostics["category_reference"]["mentions"] == 1
    assert (
        diagnostics["unsupported_social_claim_audit"]["flagged_rows"] == 1
    )


def test_age_ordered_requires_strict_two_point_gain_and_positive_ci():
    exact = select_age_label_semantics(
        opaque_variant=AGE_OPAQUE,
        ordered_variant=AGE_ORDERED,
        metrics={
            AGE_OPAQUE: {"balanced_accuracy": 0.70},
            AGE_ORDERED: {"balanced_accuracy": 0.72},
        },
        ordered_vs_opaque={"delta": 0.02, "ci_low": 0.001, "ci_high": 0.04},
    )
    above = select_age_label_semantics(
        opaque_variant=AGE_OPAQUE,
        ordered_variant=AGE_ORDERED,
        metrics={
            AGE_OPAQUE: {"balanced_accuracy": 0.70},
            AGE_ORDERED: {"balanced_accuracy": 0.721},
        },
        ordered_vs_opaque={"delta": 0.021, "ci_low": 0.001, "ci_high": 0.04},
    )
    assert exact["selected_label_semantics"] == AGE_OPAQUE
    assert above["selected_label_semantics"] == AGE_ORDERED


def test_validation_ids_are_deterministic_and_shared_by_variants():
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(500),
            "label": np.arange(500) % 2,
            "amount": (np.arange(500) % 37) + 1,
        }
    )
    # Add repeated rows so both transaction-count and volume strata vary.
    frame = pd.concat(
        [frame, frame[frame["customer_id"] % 3 == 0]], ignore_index=True
    )
    first = stratified_pilot_ids(frame)
    second = stratified_pilot_ids(frame.sample(frac=1, random_state=4))
    assert first == second
    assert len(first) == len(set(first)) == 400


def test_prompt_pilot_is_dry_run_without_api_or_files(tmp_path):
    generated = tmp_path / "generated"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_prompt_pilot.py",
            "--dataset",
            "age",
            "--model-config",
            "configs/v2/qwen.yaml",
            "--generated-dir",
            str(generated),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["mode"] == "dry-run"
    assert tuple(plan["variants"]) == PILOT_VARIANTS
    assert plan["api_requests"] == 2400
    assert not generated.exists()

"""Metrics and deterministic selection for controlled prompt-format pilots."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score


ZERO_SHOT = "guided_zero_shot_v3"
FS1 = "guided_factual_fs1_v3"
FS2 = "guided_factual_fs2_v3"
PILOT_VARIANTS = (ZERO_SHOT, FS1, FS2)


def paired_balanced_accuracy_delta(
    zero_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    samples: int = 2000,
    seed: int = 137,
) -> dict[str, float | int | str]:
    """Paired client bootstrap for BA(candidate)-BA(zero)."""
    zero = {int(row["customer_id"]): row for row in zero_rows}
    candidate = {int(row["customer_id"]): row for row in candidate_rows}
    if set(zero) != set(candidate) or not zero:
        raise ValueError("Pilot variants must contain the same non-empty client IDs")
    ordered = sorted(zero)
    truth = np.asarray([int(zero[cid]["label"]) for cid in ordered], dtype=int)
    if any(int(candidate[cid]["label"]) != int(zero[cid]["label"]) for cid in ordered):
        raise ValueError("Pilot variants disagree on validation labels")
    zero_pred = np.asarray([int(zero[cid]["predicted"]) for cid in ordered], dtype=int)
    candidate_pred = np.asarray(
        [int(candidate[cid]["predicted"]) for cid in ordered], dtype=int
    )
    point = balanced_accuracy_score(truth, candidate_pred) - balanced_accuracy_score(
        truth, zero_pred
    )
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(truth == label) for label in np.unique(truth)]
    deltas: list[float] = []
    for _ in range(int(samples)):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        deltas.append(
            balanced_accuracy_score(truth[sampled], candidate_pred[sampled])
            - balanced_accuracy_score(truth[sampled], zero_pred[sampled])
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "delta": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": 0.95,
        "method": "paired_stratified_client_bootstrap",
        "n_bootstrap": int(samples),
        "seed": int(seed),
    }


def select_prompt_variant(
    metrics: dict[str, dict[str, Any]],
    paired_deltas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered strict >2pp and positive-CI rule."""
    if ZERO_SHOT not in metrics:
        raise ValueError(f"Missing required zero-shot metrics: {ZERO_SHOT}")
    qualified: list[str] = []
    reasons: dict[str, dict[str, Any]] = {}
    zero_ba = float(metrics[ZERO_SHOT]["balanced_accuracy"])
    threshold = 0.02
    epsilon = 1e-12
    for variant in (FS1, FS2):
        if variant not in metrics or variant not in paired_deltas:
            continue
        observed_delta = float(metrics[variant]["balanced_accuracy"]) - zero_ba
        ci_low = float(paired_deltas[variant]["ci_low"])
        exceeds_threshold = observed_delta > threshold + epsilon
        passed = exceeds_threshold and ci_low > 0.0
        reasons[variant] = {
            "balanced_accuracy_delta": observed_delta,
            "paired_ci_low": ci_low,
            "strict_gain_over_2pp": exceeds_threshold,
            "paired_ci_excludes_zero": ci_low > 0.0,
            "qualified": passed,
        }
        if passed:
            qualified.append(variant)
    if not qualified:
        selected = ZERO_SHOT
    elif len(qualified) == 1:
        selected = qualified[0]
    else:
        fs1_ba = float(metrics[FS1]["balanced_accuracy"])
        fs2_ba = float(metrics[FS2]["balanced_accuracy"])
        selected = FS1 if abs(fs1_ba - fs2_ba) <= 0.005 else max(
            qualified,
            key=lambda name: float(metrics[name]["balanced_accuracy"]),
        )
    return {
        "selected_variant": selected,
        "threshold": {
            "minimum_strict_balanced_accuracy_gain": 0.02,
            "paired_ci_lower_bound_must_exceed": 0.0,
            "fs1_tie_margin": 0.005,
        },
        "candidate_decisions": reasons,
    }


def _behavioral_text(text: str) -> str:
    value = str(text or "")
    position = value.lower().rfind("final:")
    return value[:position].strip() if position >= 0 else value.strip()


def rationale_diagnostics(
    explanation_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Report diversity and a conservative rule-based social-claim audit."""
    successful = [
        row
        for row in explanation_rows
        if not row.get("error") and row.get("predicted") is not None
    ]
    rationales = [_behavioral_text(row.get("explanation", "")) for row in successful]
    normalized = [
        re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", text.lower())).strip()
        for text in rationales
    ]
    tokens = [token for text in normalized for token in text.split()]
    duplicate_rate = (
        1.0 - len(set(normalized)) / len(normalized) if normalized else 0.0
    )
    prompt_by_id = {
        int(row["customer_id"]): row for row in (prompt_rows or [])
    }
    category_mentions = 0
    clients_with_category_reference = 0
    for row, text in zip(successful, normalized):
        client_stats = str(
            prompt_by_id.get(int(row["customer_id"]), {}).get("client_stats", "")
        )
        categories = {
            match.group(1).strip().lower()
            for match in re.finditer(r"^\s*-\s*([^:]+):", client_stats, re.MULTILINE)
        }
        hits = sum(category in text for category in categories if len(category) > 2)
        category_mentions += hits
        clients_with_category_reference += int(hits > 0)
    unsupported_patterns = (
        r"\bсемь\w*",
        r"\bпрофесс\w*",
        r"\bсоциальн\w*\s+(?:роль|статус)",
        r"\bстиль\s+жизни",
        r"\bбогат\w*",
        r"\bбедн\w*",
        r"\bналичие\s+дет",
    )
    audited = [
        {
            "customer_id": int(row["customer_id"]),
            "patterns": [
                pattern
                for pattern in unsupported_patterns
                if re.search(pattern, text, re.IGNORECASE)
            ],
        }
        for row, text in zip(successful, rationales)
    ]
    flagged = [row for row in audited if row["patterns"]]
    lengths = np.asarray([len(text) for text in rationales], dtype=float)
    return {
        "n_rows": len(explanation_rows),
        "n_successful": len(successful),
        "parse_success": len(successful) / len(explanation_rows)
        if explanation_rows
        else 0.0,
        "rationale_chars": {
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "median": float(np.median(lengths)) if len(lengths) else 0.0,
            "p95": float(np.quantile(lengths, 0.95)) if len(lengths) else 0.0,
        },
        "normalized_duplicate_rate": float(duplicate_rate),
        "lexical_type_token_ratio": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "category_reference": {
            "mentions": int(category_mentions),
            "client_share": clients_with_category_reference / len(successful)
            if successful
            else 0.0,
        },
        "unsupported_social_claim_audit": {
            "flagged_rows": len(flagged),
            "flagged_share": len(flagged) / len(successful) if successful else 0.0,
            "examples": flagged[:20],
            "method": "conservative_rule_based_screen",
        },
    }

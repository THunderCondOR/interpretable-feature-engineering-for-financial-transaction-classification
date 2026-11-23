"""Helpers for extracting structured output from English LLM responses."""

from __future__ import annotations

import json
import re
import unicodedata


def extract_json_list(value: str) -> list:
    """Extract the last valid JSON array from a response."""
    if not value:
        return []
    matches = re.findall(r"\[.*?\]", value, re.DOTALL)
    greedy = re.search(r"\[.*\]", value, re.DOTALL)
    if greedy:
        matches.append(greedy.group())
    for candidate in reversed(matches):
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(result, list):
            return result
    return []


def extract_boxed_answer(value: str) -> str | None:
    """Extract the last ``\\boxed{...}`` value from a response."""
    if not value:
        return None
    matches = re.findall(r"\\boxed\{(.*?)\}", value, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def _normalize_label_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace(r"\_", "_")
    text = re.sub(r"[\s_\-\u2010-\u2015]+", " ", text)
    return text.strip(" .,:;!?'\"`|[](){}")


def normalize_text_label(value: str | None, label_names: dict | None = None) -> int | None:
    """Normalize a configured English label or a common churn synonym."""
    if value is None:
        return None
    text = _normalize_label_text(value)
    if label_names:
        for key, label in label_names.items():
            if text in {_normalize_label_text(key), _normalize_label_text(label)}:
                return int(key)
    if text in {"0", "active", "active client", "loyal", "loyal client", "retained"}:
        return 0
    if text in {"1", "churn", "churned", "churn client", "attrited", "default"}:
        return 1
    if "churn" in text or "attrit" in text or "default" in text:
        return 1
    if "active" in text or "loyal" in text or "retain" in text:
        return 0
    return None

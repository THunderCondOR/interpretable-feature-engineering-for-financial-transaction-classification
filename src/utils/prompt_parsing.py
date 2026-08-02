"""Helpers for extracting structured output from LLM responses."""

from __future__ import annotations

import json
import re
import unicodedata


def extract_json_list(s: str) -> list:
    """Extract the last valid JSON array from a string."""
    if not s:
        return []
    matches = re.findall(r"\[.*?\]", s, re.DOTALL)
    greedy = re.search(r"\[.*\]", s, re.DOTALL)
    if greedy:
        matches.append(greedy.group())
    for candidate in reversed(matches):
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            continue
    return []


def extract_boxed_answer(s: str) -> str | None:
    """Extract the last \\boxed{...} value from a string."""
    if not s:
        return None
    matches = re.findall(r"\\boxed\{(.*?)\}", s, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def _normalize_label_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace("ё", "е")
    # LaTeX boxes often escape underscores in configured labels, e.g.
    # ``no\_default``.  The escape is presentation, not class semantics.
    text = text.replace(r"\_", "_")
    text = re.sub(r"[\s_\-\u2010-\u2015]+", " ", text)
    return text.strip(" .,:;!?'\"`|[](){}")


def normalize_text_label(value: str | None, label_names: dict | None = None) -> int | None:
    """
    Normalize LLM textual answers to label ids.

    Built-in Rosbank mapping:
      0: active / активный клиент / loyal
      1: churn / отток / ушедший
    Also uses config label_names if supplied.
    """
    if value is None:
        return None
    text = _normalize_label_text(value)

    if label_names:
        for k, v in label_names.items():
            if text in {_normalize_label_text(k), _normalize_label_text(v)}:
                return int(k)

    if text in {"0", "active", "active client", "loyal", "loyal client", "активный", "активный клиент", "лояльный", "лояльный клиент"}:
        return 0
    if text in {"1", "churn", "churned", "churn client", "отток", "ушедший", "ушедший клиент", "клиент в оттоке"}:
        return 1

    if "отток" in text or "churn" in text or "ушед" in text:
        return 1
    if "актив" in text or "лоял" in text or "active" in text or "loyal" in text:
        return 0
    return None

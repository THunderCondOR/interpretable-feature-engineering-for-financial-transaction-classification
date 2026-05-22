"""
src/utils/prompt_parsing.py

Helpers for extracting structured output from LLM responses.
"""

import json
import re


def extract_json_list(s: str) -> list:
    """
    Extract the last JSON array from a string and return it as a Python list.
    Returns [] if no valid array is found.
    """
    # Find all [...] blocks (non-greedy, then greedy for nested)
    matches = re.findall(r"\[.*?\]", s, re.DOTALL)
    if not matches:
        # Try greedy match for large arrays
        match = re.search(r"\[.*\]", s, re.DOTALL)
        if match:
            matches = [match.group()]

    for candidate in reversed(matches):
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            # Try truncating to the last closing bracket
            for i in range(len(candidate), 0, -1):
                try:
                    result = json.loads(candidate[:i])
                    if isinstance(result, list):
                        return result
                except Exception:
                    continue
    return []


def extract_boxed_answer(s: str) -> str | None:
    """
    Extract the last \\boxed{...} value from a string.
    Returns None if not found.
    """
    matches = re.findall(r"\\boxed\{(.*?)\}", s)
    return matches[-1].strip() if matches else None
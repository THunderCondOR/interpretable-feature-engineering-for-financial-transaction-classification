from __future__ import annotations

from interpretable_features.analysis.judge_grounding import (
    make_dialogue,
    parse_json_object,
    validate_judgment,
)


def test_grounding_dialogue_blinds_target_and_prediction():
    dialogue = make_dialogue(
        {
            "customer_id": "client-1",
            "client_stats": "Five grocery payments.",
            "train_reference_summary": "Class summaries.",
            "field_semantics": "Amounts are unsigned values.",
            "claim": "The client purchases groceries repeatedly.",
            "label": 1,
            "prediction": 0,
        }
    )
    rendered = "\n".join(message["content"] for message in dialogue)
    assert "Five grocery payments" in rendered
    assert "prediction" not in rendered.lower()
    assert "client-1" not in rendered
    assert '"label"' not in rendered.lower()


def test_grounding_parser_requires_contract():
    parsed, error = parse_json_object('{"verdict":"supported","evidence":"five payments"}')
    assert error is None
    assert parsed["verdict"] == "supported"
    assert validate_judgment({"verdict": "yes", "reason": "No."})

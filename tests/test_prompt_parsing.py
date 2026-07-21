from src.utils.prompt_parsing import normalize_text_label


AGE_LABELS = {
    "0": "18-25",
    "1": "26-35",
    "2": "36-45",
    "3": "46+",
}


def test_age_labels_with_hyphens_are_normalized_symmetrically() -> None:
    assert normalize_text_label("18-25", AGE_LABELS) == 0
    assert normalize_text_label("26–35", AGE_LABELS) == 1
    assert normalize_text_label("36 - 45", AGE_LABELS) == 2
    assert normalize_text_label("46+", AGE_LABELS) == 3

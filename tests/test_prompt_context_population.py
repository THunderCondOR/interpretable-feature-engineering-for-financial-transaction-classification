import pandas as pd

import run_pipeline


def test_prompt_context_uses_complete_train_not_target_subset(monkeypatch):
    calls = []
    complete_train = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "label": [0, 1, 1],
        }
    )

    def fake_load_split(config, split, *, apply_client_filter=True):
        calls.append((split, apply_client_filter))
        return complete_train

    monkeypatch.setattr(run_pipeline, "load_split", fake_load_split)
    config = {
        "pipeline": {
            "prompt_context_split": "train",
            "prompt_context_population": "full_train_split",
        }
    }

    observed = run_pipeline.load_prompt_context(config)

    assert observed.equals(complete_train)
    assert calls == [("train", False)]


def test_prompt_context_rejects_ambiguous_subset_policy():
    config = {
        "pipeline": {
            "prompt_context_split": "train",
            "prompt_context_population": "target_subset",
        }
    }

    try:
        run_pipeline.load_prompt_context(config)
    except ValueError as exc:
        assert "complete train split" in str(exc)
    else:
        raise AssertionError("subset prompt context must be rejected")

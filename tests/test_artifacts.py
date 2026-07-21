import json

import pytest

from src.experiments.artifacts import (
    ensure_run_manifest,
    fingerprint,
    prompt_signature,
)


def test_fingerprint_is_order_independent_for_mappings() -> None:
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_prompt_signature_changes_with_prompt_or_decoding() -> None:
    base = {
        "system_prompt": "system",
        "user_prompt": "client A",
        "model": "model",
        "decoding": {"temperature": 0.8, "seed": 17},
    }
    signature = prompt_signature(**base)
    assert signature != prompt_signature(**{**base, "user_prompt": "client B"})
    assert signature != prompt_signature(
        **{**base, "decoding": {"temperature": 0.8, "seed": 101}}
    )


def test_manifest_prevents_reusing_output_for_incompatible_config(tmp_path) -> None:
    data = tmp_path / "train.csv"
    data.write_text("customer_id,label\n1,0\n", encoding="utf-8")
    prompt = tmp_path / "system.txt"
    prompt.write_text("system", encoding="utf-8")
    out = tmp_path / "out"
    config = {
        "experiment": {"run_id": "test", "variant": "v1", "seed": 17},
        "dataset": {"name": "test", "splits": {"train": str(data)}},
        "prompts": {"base_dir": str(tmp_path), "system": "system.txt"},
        "output": {"base_dir": str(out)},
    }

    path = ensure_run_manifest(config, repo_root=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["dataset_files"][str(data)]["sha256"]
    assert ensure_run_manifest(config, repo_root=tmp_path) == path

    incompatible = {**config, "experiment": {**config["experiment"], "seed": 101}}
    with pytest.raises(RuntimeError, match="incompatible experiment"):
        ensure_run_manifest(incompatible, repo_root=tmp_path)

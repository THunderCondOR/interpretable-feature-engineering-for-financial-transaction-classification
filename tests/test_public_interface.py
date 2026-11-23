from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from interpretable_features.cli import main
from interpretable_features.data.prompt_locale import assert_english_model_text


def test_cli_help(capsys):
    main(["--help"])
    assert "grounding-judge" in capsys.readouterr().out


def test_grounding_judge_is_dry_run_by_default(tmp_path: Path, capsys):
    main(
        [
            "grounding-judge",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--model-config",
            str(tmp_path / "missing.yaml"),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--judge-name",
            "judge_a",
        ]
    )
    assert '"mode": "dry-run"' in capsys.readouterr().out
    assert not (tmp_path / "output.jsonl").exists()


def test_fidelity_cli(tmp_path: Path):
    source = tmp_path / "probabilities.npz"
    output = tmp_path / "fidelity.json"
    np.savez(
        source,
        teacher_probabilities=np.array([[0.9, 0.1], [0.2, 0.8]]),
        surrogate_probabilities=np.array([[0.8, 0.2], [0.3, 0.7]]),
        labels=np.array([0, 1]),
    )
    main(["fidelity", "--input", str(source), "--output", str(output)])
    assert json.loads(output.read_text())["hard_agreement"] == 1.0


def test_stability_cli(tmp_path: Path):
    source = tmp_path / "assignments.csv"
    output = tmp_path / "stability.csv"
    pd.DataFrame(
        {
            "anchor_id": [1, 2, 3, 1, 2, 3],
            "run": ["a", "a", "a", "b", "b", "b"],
            "cluster_id": [0, 0, 1, 4, 4, 8],
        }
    ).to_csv(source, index=False)
    main(
        ["stability", "--input", str(source), "--axis", "generation_seed", "--output", str(output)]
    )
    result = pd.read_csv(output)
    assert result.loc[0, "ari"] == 1.0
    assert result.loc[0, "nmi"] == 1.0


def test_english_boundary_rejects_non_english_text():
    try:
        assert_english_model_text(
            "\u041d\u0435-\u0430\u043d\u0433\u043b\u0438\u0439\u0441\u043a\u0438\u0439 \u0442\u0435\u043a\u0441\u0442",
            context="test",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected non-English text to be rejected")


def test_paper_profiles_separate_generation_from_extraction():
    from interpretable_features.runner import load_config

    for path in Path("configs/datasets").glob("*.yaml"):
        profile = load_config(path, "configs/models/gpt_oss.yaml")
        assert profile["llm"]["temperature"] == 0.8
        assert profile["claims_generation"]["temperature"] == 0
        assert profile["claims_generation"]["max_tokens"] == 2048
        assert profile["clustering"]["min_client_coverage"] == 5
        assert "extra_body" not in profile["llm"]


def test_prepared_benchmark_roles_are_disjoint(tmp_path, monkeypatch):
    from interpretable_features.runner import load_config, load_prompt_context, load_split

    config = load_config("configs/datasets/cofinfad.yaml")
    monkeypatch.chdir(tmp_path)
    events = Path(config["dataset"]["splits"]["train"])
    events.parent.mkdir(parents=True)
    records = []
    for label, (split, ids) in enumerate(config["dataset"]["client_ids_by_split"].items()):
        Path(ids).write_text(json.dumps([split]))
        records.append(
            {
                "customer_id": split,
                "label": label,
                "tr_datetime": "2026-01-01",
                "amount": 2.0,
                "mcc_code_desc": "transfer",
            }
        )
    pd.DataFrame(records).to_parquet(events)
    assert set(load_prompt_context(config)["customer_id"]) == {"train"}
    assert set(load_split(config, "test")["customer_id"]) == {"test"}


def test_report_escapes_run_name_and_embeds_the_bundle():
    from interpretable_features.reporting.analysis_report import build_report, synthetic_bundle

    document = build_report(synthetic_bundle(), "run <one>")
    assert "run &lt;one&gt;" in document
    assert 'id="report-data"' in document

import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_gender_v2 import (
    FULL_API_STEPS,
    FULL_SPLITS,
    _metric,
    _write_full_completion,
    _write_pilot_completion,
    materialize,
    select_variant,
)
from src.experiments.artifacts import file_sha256, fingerprint
from src.experiments.config_builder import load_yaml


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _base_config(tmp_path: Path, *, n_clients: int = 400) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = pd.DataFrame(
        {
            "customer_id": list(range(n_clients)),
            "tr_datetime": ["2024-01-01"] * n_clients,
            "amount": [float(index + 1) for index in range(n_clients)],
            "mcc_code_desc": ["category"] * n_clients,
            "label": [index % 2 for index in range(n_clients)],
        }
    )
    split_paths = {}
    for split in FULL_SPLITS:
        path = data_dir / f"{split}.csv"
        rows.to_csv(path, index=False)
        split_paths[split] = str(path)
    legacy_dir = tmp_path / "legacy"
    _write_jsonl(
        legacy_dir / "explanations_val.jsonl",
        [
            {
                "customer_id": index,
                "label": index % 2,
                "predicted": index % 2,
                "explanation": "Observed transaction behaviour. Final: label.",
                "error": None,
            }
            for index in range(n_clients)
        ],
    )
    return _write_yaml(
        tmp_path / "gender.yaml",
        {
            "dataset": {
                "name": "gender",
                "columns": {
                    "customer_id": "customer_id",
                    "datetime": "tr_datetime",
                    "amount": "amount",
                    "category": "mcc_code_desc",
                    "label": "label",
                },
                "splits": split_paths,
                "label_names": {"0": "zero", "1": "one"},
            },
            "llm": {"api_base_url": "unused", "api_key": "unused"},
            "prompts": {
                "base_dir": str(tmp_path),
                "system": "system.txt",
                "user": "user.txt",
                "claims_system": "claims_system.txt",
                "claims_user": "claims_user.txt",
            },
            "pipeline": {"n_explanation_samples": 1, "n_claims_samples": 1},
            "output": {
                "base_dir": str(legacy_dir),
                "clients_stats": "clients_stats.jsonl",
                "prompts": "prompts.jsonl",
                "explanations": "explanations.jsonl",
                "claims": "claims.jsonl",
            },
        },
    )


def _model_profile(tmp_path: Path, slug: str) -> Path:
    return _write_yaml(
        tmp_path / f"{slug}.yaml",
        {
            "experiment": {"model_slug": slug, "seed": 17},
            "generation": {
                "model": f"test/{slug}",
                "temperature": 0.8,
                "top_p": 0.9,
                "seed": 17,
            },
            "claims_generation": {
                "model": f"test/{slug}",
                "temperature": 0.0,
                "seed": 17,
            },
            "execution": {"initial_concurrency": 64},
        },
    )


def _complete_outputs(
    config: dict, ids_by_split: dict[str, list[int]], *, claims: bool
) -> None:
    out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in ids_by_split.items():
        _write_jsonl(
            out_dir / f"clients_stats_{split}.jsonl",
            [{"customer_id": cid} for cid in ids],
        )
        prompt_rows = [
            {"customer_id": cid, "prompt_hash": f"prompt-{split}-{cid}"}
            for cid in ids
        ]
        _write_jsonl(out_dir / f"prompts_{split}.jsonl", prompt_rows)
        _write_jsonl(
            out_dir / f"explanations_{split}.jsonl",
            [
                {
                    "customer_id": cid,
                    "label": cid % 2,
                    "predicted": cid % 2,
                    "explanation": "Observed transaction behaviour. Final: label.",
                    "error": None,
                    "error_type": None,
                    "generation_signature": f"generation-{split}-{cid}",
                    "prompt_hash": f"prompt-{split}-{cid}",
                }
                for cid in ids
            ],
        )
        (out_dir / f"llm_metrics_{split}.json").write_text(
            json.dumps({
                "split": split,
                "n_rows": len(ids),
                "n_scored": len(ids),
                "n_skipped": 0,
                "n_errors": 0,
                "coverage": 1.0,
            }),
            encoding="utf-8",
        )
        if claims:
            _write_jsonl(
                out_dir / f"claims_{split}.jsonl",
                [
                    {
                        "customer_id": cid,
                        "claims": ["Observed claim"],
                        "error": None,
                        "error_type": None,
                        "generation_signature": f"claims-{split}-{cid}",
                        "source_explanation_hash": f"source-{split}-{cid}",
                        "source_prompt_hashes": [f"prompt-{split}-{cid}"],
                    }
                    for cid in ids
                ],
            )
    (out_dir / "manifest.json").write_text(
        json.dumps({"manifest_sha256": fingerprint(config)}), encoding="utf-8"
    )


def test_metric_reconstructs_subset_when_aggregate_metrics_are_absent(tmp_path):
    _write_jsonl(
        tmp_path / "explanations_val.jsonl",
        [
            {"customer_id": 1, "label": 0, "predicted": 0, "error": None},
            {"customer_id": 2, "label": 1, "predicted": 1, "error": None},
            {"customer_id": 3, "label": 1, "predicted": 0, "error": None},
        ],
    )
    result = _metric(tmp_path / "llm_metrics_val.json", [1, 2])
    assert result["n_rows"] == 2
    assert result["balanced_accuracy"] == 1.0


def test_materialize_pilot_and_selection_are_content_addressed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base_path = _base_config(tmp_path)
    qwen_path = _model_profile(tmp_path, "qwen")
    gpt_path = _model_profile(tmp_path, "gpt_oss")
    generated = tmp_path / "logs/runs/night/generated"
    materialized = materialize(
        run_id="night",
        base_config_path=base_path,
        model_config_path=qwen_path,
        generated_dir=generated,
        results_root=tmp_path / "results/v2",
    )

    pilot_ids = json.loads(Path(materialized["pilot_ids"]).read_text())
    assert len(pilot_ids) == len(set(pilot_ids)) == 400
    assert materialized["sampling_seed"] == 137
    assert materialized["generation_seed"] == 17
    assert materialized["pilot_ids_sha256"] == file_sha256(materialized["pilot_ids"])
    assert materialized["legacy_subset_metrics"]["n_rows"] == 400
    for variant, config_path in materialized["pilot_configs"].items():
        config = load_yaml(config_path)
        assert config["experiment"]["seed"] == 17
        assert config["experiment"]["sampling_seed"] == 137
        assert config["generation"]["seed"] == 17
        assert materialized["pilot_config_sha256"][variant] == file_sha256(config_path)
        _complete_outputs(config, {"val": pilot_ids}, claims=False)

    marker_path = _write_pilot_completion(materialized)
    marker = json.loads(marker_path.read_text())
    assert marker["status"] == "completed"
    assert marker["expected_counts"] == {"val": 400}

    selection = select_variant(
        materialized,
        base_config_path=base_path,
        model_config_path=qwen_path,
        gpt_model_config_path=gpt_path,
        generated_dir=generated,
        results_root=tmp_path / "results/v2",
    )
    assert selection["selected_variant"] == "neutral_robust_zero_shot"
    assert selection["metrics_sha256"] == fingerprint(selection["metrics"])
    assert set(selection["selected_configs"]) == {"qwen", "gpt_oss"}
    for entry in selection["selected_configs"].values():
        assert entry["sha256"] == file_sha256(entry["path"])
    unsigned = {key: value for key, value in selection.items() if key != "selection_sha256"}
    assert selection["selection_sha256"] == fingerprint(unsigned)


def test_full_completion_requires_all_three_explanation_and_claim_splits(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    base_path = _base_config(tmp_path, n_clients=4)
    profile_path = _model_profile(tmp_path, "qwen")
    from src.experiments.config_builder import build_runtime_config

    config = build_runtime_config(
        load_yaml(base_path),
        load_yaml(profile_path),
        run_id="night",
        variant="neutral_robust_zero_shot",
        seed=17,
        results_root=tmp_path / "results/v2",
    )
    config_path = _write_yaml(tmp_path / "selected.yaml", config)
    ids_by_split = {split: [0, 1, 2, 3] for split in FULL_SPLITS}
    _complete_outputs(config, ids_by_split, claims=True)
    selection = {
        "selected_variant": "neutral_robust_zero_shot",
        "selection_sha256": "selection-hash",
    }
    marker_path = _write_full_completion(
        run_id="night",
        selection=selection,
        config=config,
        config_path=config_path,
    )
    marker = json.loads(marker_path.read_text())
    assert marker["status"] == "completed"
    assert marker["splits"] == ["train", "val", "test"]
    assert marker["expected_counts"] == {"train": 4, "val": 4, "test": 4}
    assert FULL_API_STEPS == "stats,prompts,cot,llm_eval,claims"

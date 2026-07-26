import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.benchmark_registry import (
    BERKA_SEEDS,
    berka_folds,
    datafusion_folds,
    normalize_datafusion,
    normalize_berka,
)
from src.data.loader import load_dataset
from src.evaluation.prompt_pilot import (
    paired_primary_metric_delta,
    select_prompt_variant_by_metric,
)
from src.experiments.artifacts import atomic_write_json, fingerprint
from src.experiments.config_builder import load_yaml
from src.experiments.cv_config import build_cv_runtime_config
from scripts.cv_preflight import validate_prepared


ROOT = Path(__file__).parents[1]


def test_berka_normalization_is_strictly_preloan_and_removes_uver(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "loan_id": [1, 2, 3, 4],
            "account_id": [11, 12, 13, 14],
            "date": ["1996-01-10"] * 4,
            "amount": [100] * 4,
            "duration": [12] * 4,
            "payments": [10] * 4,
            "status": ["A", "B", "C", "D"],
        }
    ).to_csv(raw / "loan.csv", index=False)
    rows = []
    for account_id in (11, 12, 13, 14):
        rows.extend(
            [
                {
                    "trans_id": account_id * 10,
                    "account_id": account_id,
                    "date": "1996-01-01",
                    "type": "PRIJEM",
                    "operation": "VKLAD",
                    "amount": 100,
                    "balance": 100,
                    "k_symbol": "",
                    "bank": "",
                    "account": "",
                },
                {
                    "trans_id": account_id * 10 + 1,
                    "account_id": account_id,
                    "date": "1996-01-02",
                    "type": "VYDAJ",
                    "operation": "PREVOD NA UCET",
                    "amount": 20,
                    "balance": 80,
                    "k_symbol": "UVER",
                    "bank": "",
                    "account": "",
                },
                {
                    "trans_id": account_id * 10 + 2,
                    "account_id": account_id,
                    "date": "1996-01-10",
                    "type": "VYDAJ",
                    "operation": "VYBER",
                    "amount": 30,
                    "balance": 50,
                    "k_symbol": "",
                    "bank": "",
                    "account": "",
                },
            ]
        )
    pd.DataFrame(rows).to_csv(raw / "trans.csv", index=False)
    events, labels, _ = normalize_berka(raw)
    assert len(events) == 4
    assert (events["tr_datetime"] < pd.Timestamp("1996-01-10")).all()
    assert set(labels.loc[labels["label"] == 1, "customer_id"]) == {"2", "4"}
    assert events["amount"].gt(0).all()
    assert events["mcc_code_desc"].str.contains("cash deposit").all()


def test_berka_folds_have_published_sizes_and_are_reproducible():
    labels = pd.DataFrame({
        "customer_id": [str(value) for value in range(682)],
        "label": [0] * 606 + [1] * 76,
    })
    first = berka_folds(labels)
    second = berka_folds(labels.sample(frac=1, random_state=44))
    assert first == second
    assert len(first) == len(BERKA_SEEDS) == 5
    for train, test in first:
        assert len(train) == 478
        assert len(test) == 204
        assert not set(train) & set(test)
        assert len(set(train) | set(test)) == 682


def test_datafusion_approximate_folds_are_disjoint_and_complete():
    labels = pd.DataFrame({
        "customer_id": [f"hex-{value:04x}" for value in range(101)],
        "label": np.arange(101) % 2,
    })
    folds = datafusion_folds(labels, backend="sklearn_approx")
    for train, test in folds:
        assert not set(train) & set(test)
        assert len(set(train) | set(test)) == 101
    assert set().union(*(set(test) for _, test in folds)) == set(
        labels["customer_id"]
    )


def test_datafusion_streaming_normalizer_preserves_opaque_ids(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({
        "bank": ["a0ff", "b100"],
        "higher_education": [0, 1],
    }).to_csv(raw / "train.csv", index=False)
    pd.DataFrame({
        "user_id": ["a0ff", "b100"],
        "mcc_code": [5411, 9999],
        "currency_rk": [643, 643],
        "transaction_amt": [-12.0, 30.0],
        "transaction_dttm": ["2020-01-01", "2020-01-02"],
    }).to_csv(raw / "transactions.csv", index=False)
    pd.DataFrame({
        "currency_rk": [643],
        "Name": ["RUR"],
    }).to_csv(raw / "currency_rk.csv", index=False)
    output = tmp_path / "events.parquet"
    labels, _ = normalize_datafusion(raw, output)
    events = pd.read_parquet(output)
    assert labels["customer_id"].tolist() == ["a0ff", "b100"]
    assert events["customer_id"].tolist() == ["a0ff", "b100"]
    assert events["mcc_code_desc"].tolist() == ["Supermarkets", "MCC 9999"]
    assert events["currency_name"].tolist() == ["RUR", "RUR"]


def test_parquet_loader_pushes_string_id_filter(tmp_path):
    path = tmp_path / "events.parquet"
    pd.DataFrame({
        "customer_id": ["a1", "b2"],
        "tr_datetime": ["2020-01-01", "2020-01-02"],
        "amount": [-10.0, 20.0],
        "mcc_code_desc": ["Books", "Taxi"],
        "label": [0, 1],
    }).to_parquet(path, index=False)
    config = {
        "dataset": {
            "splits": {"train": str(path)},
            "client_ids_by_split": {"train": ["b2"]},
            "columns": {
                "customer_id": "customer_id",
                "datetime": "tr_datetime",
                "amount": "amount",
                "category": "mcc_code_desc",
                "label": "label",
            },
        }
    }
    loaded = load_dataset(config, "train")
    assert loaded["customer_id"].tolist() == ["b2"]


def _pilot_rows(predictions):
    return [
        {"customer_id": f"id-{index}", "label": index % 2, "predicted": value}
        for index, value in enumerate(predictions)
    ]


def test_v5_prompt_selection_uses_primary_metric_and_strict_threshold():
    truth = [index % 2 for index in range(100)]
    zero = _pilot_rows([0] * 100)
    strong = _pilot_rows(truth)
    delta = paired_primary_metric_delta(
        zero, strong, metric="roc_auc", samples=100, seed=7
    )
    metrics = {
        "guided_zero_shot_v5": {"roc_auc": 0.50},
        "guided_factual_fs1_v5": {"roc_auc": 1.00},
        "guided_factual_fs2_v5": {"roc_auc": 0.50},
    }
    decision = select_prompt_variant_by_metric(
        metrics,
        {
            "guided_factual_fs1_v5": delta,
            "guided_factual_fs2_v5": {
                "ci_low": -0.01,
            },
        },
        metric="roc_auc",
    )
    assert decision["selected_variant"] == "guided_factual_fs1_v5"


def test_cv_config_binds_pilot_to_inner_train_only(tmp_path):
    prepared = tmp_path / "prepared"
    fold_root = prepared / "berka" / "unittab_70_30_5seed" / "fold_0"
    fold_root.mkdir(parents=True)
    events = fold_root.parent / "events.parquet"
    events.write_bytes(b"fixture")
    ids = {
        "outer_train": ["1", "2", "3"],
        "outer_test": ["4"],
        "inner_train": ["1", "2"],
        "inner_validation": ["3"],
    }
    for role, values in ids.items():
        atomic_write_json(fold_root / f"{role}_ids.json", values)
    fold = {
        "dataset": "berka",
        "protocol": "unittab_70_30_5seed",
        "fold": 0,
        "fold_signature": fingerprint(ids),
        "counts": {role: len(values) for role, values in ids.items()},
    }
    atomic_write_json(fold_root / "fold_manifest.json", fold)
    benchmark = {"events": str(events)}
    atomic_write_json(fold_root.parent / "benchmark_manifest.json", benchmark)
    config = build_cv_runtime_config(
        load_yaml("configs/v5/berka.yaml"),
        load_yaml("configs/v2/qwen.yaml"),
        run_id="test-v5",
        variant="guided_zero_shot_v5",
        fold_manifest_path=fold_root / "fold_manifest.json",
        fold_manifest=fold,
        benchmark_manifest_path=fold_root.parent / "benchmark_manifest.json",
        results_root=tmp_path / "results",
        mode="pilot",
    )
    assert config["cv"]["split_roles"] == {
        "train": "inner_train",
        "val": "inner_validation",
    }
    assert config["pipeline"]["prompt_context_apply_client_filter"] is True
    assert config["dataset"]["client_ids_by_split"]["train"].endswith(
        "inner_train_ids.json"
    )


def test_v5_runners_are_dry_run_without_artifacts(tmp_path):
    for command in (
        [
            "scripts/prepare_benchmark_dataset.py",
            "--dataset", "berka",
            "--output-root", str(tmp_path / "prepared"),
        ],
        [
            "scripts/run_cv_prompt_pilots.py",
            "--dataset", "berka",
            "--generated-root", str(tmp_path / "generated"),
        ],
        [
            "scripts/run_cv_llm_queue.py",
            "--model", "qwen",
            "--run-id", "dry-v5",
        ],
        [
            "scripts/run_cv_offline_pipeline.py",
            "--datasets", "berka",
            "--models", "qwen",
            "--folds", "0",
        ],
        ["scripts/cv_preflight.py"],
        [
            "scripts/cv_api_probe.py",
            "--model-config", "configs/v2/qwen.yaml",
            "--model-config", "configs/v2/gpt_oss.yaml",
        ],
        ["scripts/summarize_cv_benchmarks.py"],
    ):
        result = subprocess.run(
            [sys.executable, *command],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(result.stdout)["mode"] == "dry-run"
    assert not (tmp_path / "prepared").exists()
    assert not (tmp_path / "generated").exists()


def test_paid_preflight_rejects_approximate_datafusion_folds(tmp_path):
    prepared = tmp_path / "prepared"
    for dataset, protocol, count, backend in (
        ("berka", "unittab_70_30_5seed", 682, "numpy_random_state_protocol_match"),
        ("datafusion_education", "mbd_5fold_seed42", 8509, "sklearn_approx"),
    ):
        root = prepared / dataset / protocol
        root.mkdir(parents=True)
        atomic_write_json(
            root / "benchmark_manifest.json",
            {"benchmark": {"n_entities": count}},
        )
        for fold in range(5):
            fold_root = root / f"fold_{fold}"
            fold_root.mkdir()
            atomic_write_json(
                fold_root / "fold_manifest.json",
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "fold": fold,
                    "split_backend": backend,
                    "ids": {
                        "outer_train": ["train"],
                        "outer_test": ["test"],
                        "inner_train": ["inner-train"],
                        "inner_validation": ["inner-validation"],
                    },
                    "counts": {},
                },
            )
    with pytest.raises(RuntimeError, match="pyspark==3.3.3"):
        validate_prepared(prepared)

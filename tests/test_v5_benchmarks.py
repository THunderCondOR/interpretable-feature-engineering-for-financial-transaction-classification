import hashlib
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
from src.data.profiles import format_client_profile
from src.pipeline.llm_eval import summarize_prediction_rows
from src.evaluation.prompt_pilot import (
    paired_primary_metric_delta,
    select_prompt_variant_by_metric,
)
from src.experiments.artifacts import atomic_write_json, fingerprint
from src.experiments.config_builder import load_yaml
from src.experiments.cv_config import build_cv_runtime_config
from scripts.cv_preflight import validate_prepared
from scripts.run_cv_llm_queue import (
    jobs as cv_queue_jobs,
    valid_output as valid_cv_output,
)


ROOT = Path(__file__).parents[1]


def test_gpt_pilot_selector_does_not_replace_full_qwen_profile():
    queue = cv_queue_jobs(
        model="gpt_oss",
        run_id="selector-profile-test",
        qwen_config=Path("configs/v2/qwen.yaml"),
        gpt_config=Path("configs/v2/gpt_oss.yaml"),
        datasets=("berka",),
        run_pilots=True,
    )
    pilot = queue[0]["command"]
    assert pilot[pilot.index("--pilot-config") + 1] == "configs/v2/gpt_oss.yaml"
    assert pilot[pilot.index("--qwen-config") + 1] == "configs/v2/qwen.yaml"
    assert pilot[pilot.index("--gpt-config") + 1] == "configs/v2/gpt_oss.yaml"


def test_default_v5_queues_run_qwen_pilots_but_not_gpt_pilots():
    qwen = cv_queue_jobs(
        model="qwen",
        run_id="parallel-test",
        qwen_config=Path("configs/v2/qwen.yaml"),
        gpt_config=Path("configs/v2/gpt_oss.yaml"),
    )
    gpt = cv_queue_jobs(
        model="gpt_oss",
        run_id="parallel-test",
        qwen_config=Path("configs/v2/qwen.yaml"),
        gpt_config=Path("configs/v2/gpt_oss.yaml"),
    )
    assert any(job["kind"] == "pilot" for job in qwen)
    assert all(job["kind"] == "full" for job in gpt)
    assert [job["dataset"] for job in qwen].index(
        "datafusion_education"
    ) > [job["dataset"] for job in qwen].index("berka")


def test_datafusion_only_queue_has_exact_request_budget():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_cv_llm_queue.py",
            "--model", "qwen",
            "--run-id", "budget-test",
            "--datasets", "datafusion_education",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["main_api_requests_both_models"] == 170_180
    assert plan["pilot_requests_qwen"] == 6_000


def test_cv_completion_reuse_rejects_wrong_model(tmp_path):
    path = tmp_path / "completion.json"
    split_evidence = {}
    for split in ("train", "test"):
        artifacts = {}
        for index in range(5):
            artifact = tmp_path / f"{split}_{index}.json"
            artifact.write_text(f"{split}-{index}", encoding="utf-8")
            artifacts[str(artifact)] = {
                "exists": True,
                "size": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        split_evidence[split] = {
            "expected_clients": 1,
            "artifacts": artifacts,
        }
    identity = {
        "run_id": "run",
        "dataset": "berka",
        "model_slug": "qwen",
        "variant": "guided_zero_shot_v5",
        "protocol": "unittab_70_30_5seed",
        "fold": 0,
        "split_evidence": split_evidence,
    }
    path.write_text(
        json.dumps({
            "status": "completed",
            **identity,
            "completion_signature": fingerprint(identity),
            "completed_at": "now",
        }),
        encoding="utf-8",
    )
    assert valid_cv_output(
        path,
        run_id="run",
        job={
            "kind": "full",
            "dataset": "berka",
            "fold": 0,
            "model": "qwen",
        },
    )
    assert not valid_cv_output(
        path,
        run_id="run",
        job={
            "kind": "full",
            "dataset": "berka",
            "fold": 0,
            "model": "gpt_oss",
        },
    )


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


def test_datafusion_public_folds_match_sklearn_kfold_on_original_row_order():
    labels = pd.DataFrame({
        "customer_id": [f"id-{value}" for value in [8, 2, 9, 1, 7, 3, 6, 4, 5, 0]],
        "label": np.arange(10) % 2,
    })
    expected_tests = []
    from sklearn.model_selection import KFold

    for _, test_index in KFold(
        n_splits=5, shuffle=True, random_state=100
    ).split(labels):
        expected_tests.append(set(labels.iloc[test_index]["customer_id"]))
    actual = datafusion_folds(labels, backend="sklearn_public")
    assert [set(test) for _, test in actual] == expected_tests
    assert set().union(*(set(test) for _, test in actual)) == set(
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


def test_categorical_top_k_never_includes_unused_zero_categories():
    frame = pd.DataFrame({
        "customer_id": ["1", "1", "1"],
        "tr_datetime": pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03"]
        ),
        "amount": [10.0, -2.0, -3.0],
        "mcc_code_desc": pd.Categorical(
            ["Books", "Books", "Taxi"],
            categories=["Books", "Taxi", "Unused"],
        ),
        "label": [0, 0, 0],
        "operation_type": ["credit", "debit", "debit"],
        "balance": [10.0, 8.0, 5.0],
    })
    profile = format_client_profile(frame, load_yaml("configs/v5/berka.yaml"))
    category_block = profile.split(
        "* Observed top-2 operation and payment-purpose categories:", 1
    )[1].split("* Coverage note", 1)[0]
    assert "Unused" not in category_block
    assert "0 transactions" not in category_block
    assert category_block.count("\n  - ") == 2


def test_datafusion_profile_never_sums_different_currencies():
    frame = pd.DataFrame({
        "customer_id": ["client"] * 4,
        "tr_datetime": pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
        ),
        "amount": [100.0, -50.0, 7.0, -3.0],
        "mcc_code_desc": ["A", "B", "C", "D"],
        "currency_name": ["RUR", "RUR", "USD", "USD"],
        "label": [0, 0, 0, 0],
    })
    profile = format_client_profile(
        frame, load_yaml("configs/v5/datafusion_education.yaml")
    )
    assert "amounts are never summed across currencies" in profile
    assert "RUR: 2 operations" in profile
    assert "total inflow=100.00" in profile
    assert "USD: 2 operations" in profile
    assert "total inflow=7.00" in profile
    assert "Total inflow: 107.00" not in profile


def test_direct_binary_metrics_do_not_publish_probability_roc_auc():
    metrics = summarize_prediction_rows(
        [
            {"label": 0, "predicted": 0, "error": None},
            {"label": 1, "predicted": 1, "error": None},
        ],
        split="test",
    )
    assert "roc_auc" not in metrics
    assert metrics["hard_label_auc"] == 1.0
    assert "not comparable" in metrics["hard_label_auc_note"]


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
    assert config["llm"]["api_base_url"] == "${API_BASE_URL}"
    assert config["llm"]["api_key"] == "${API_KEY}"
    assert config["dataset"]["client_ids_by_split"]["train"].endswith(
        "inner_train_ids.json"
    )
    assert "/runs/test-v5/" in config["output"]["base_dir"]


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
        ("datafusion_education", "public_kfold5_seed100", 8509, "sklearn_approx"),
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
    with pytest.raises(RuntimeError, match="public notebook folds"):
        validate_prepared(prepared)

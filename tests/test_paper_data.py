import json
from pathlib import Path

import pandas as pd

from scripts.build_paper_data import (
    Snapshot,
    aggregate_fold_metrics,
    article_deltas,
    build_architecture_doc,
    canonical_cells,
    valid_completion,
)


def test_snapshot_records_source_hash_and_jsonl_rows(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")
    output = tmp_path / "paper_data"
    snapshot = Snapshot(output)
    target = snapshot.copy(source, Path("claims/test.jsonl"), role="claims")
    assert target.read_text() == source.read_text()
    assert snapshot.files[0]["rows"] == 2
    assert len(snapshot.files[0]["sha256"]) == 64


def test_completion_rejects_partial_or_wrong_identity(tmp_path):
    artifact = tmp_path / "claims.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps({
        "status": "completed", "dataset": "berka", "model_slug": "qwen",
        "fold": 0, "completion_signature": "sig",
        "split_evidence": {"test": {"artifacts": {
            str(artifact): {"size": artifact.stat().st_size}
        }}},
    }))
    assert valid_completion(marker, dataset="berka", model="qwen", fold=0)
    assert not valid_completion(marker, dataset="berka", model="gpt_oss", fold=0)


def test_article_delta_is_recomputed_from_current_accuracy():
    metrics = pd.DataFrame([{
        "dataset": "gender", "model": "qwen", "protocol": "v4",
        "method": "cot", "classifier": "xgboost", "accuracy": .70,
    }])
    delta = article_deltas(metrics).iloc[0]
    assert delta.current_accuracy == .70
    assert abs(delta.delta_current_minus_submitted - (.70 - .683)) < 1e-12


def test_fold_aggregation_uses_sample_sd():
    frame = pd.DataFrame([
        {"dataset": "berka", "model": "qwen", "protocol": "p", "method": "cot", "classifier": "xgboost", "accuracy": .6},
        {"dataset": "berka", "model": "qwen", "protocol": "p", "method": "cot", "classifier": "xgboost", "accuracy": .8},
    ])
    result = aggregate_fold_metrics(frame).iloc[0]
    assert result.n_cells == 2
    assert result.accuracy == .7
    assert result.accuracy_sd > 0


def test_architecture_documents_train_only_boundary_and_repair_queue():
    text = build_architecture_doc()
    assert "Validation/test claims" in text
    assert "repair queue" in text
    assert "process lease" in text


def test_current_registry_does_not_treat_partial_datafusion_as_complete():
    cells, pending = canonical_cells()
    assert any(cell.dataset == "berka" for cell in cells)
    if not all(
        Path(f"logs/runs/reviewer-v10-datafusion-public/completion/{model}_datafusion_education_dataset.json").is_file()
        for model in ("qwen", "gpt_oss")
    ):
        assert not any(cell.dataset == "datafusion_education" for cell in cells)
        assert any("datafusion_education" in row["experiment"] for row in pending)

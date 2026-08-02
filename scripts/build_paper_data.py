#!/usr/bin/env python3
"""Build a portable, reviewer-facing snapshot of completed experiments."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Cell:
    dataset: str
    model: str
    protocol: str
    source_root: Path
    derived_root: Path | None
    fold: int | None = None
    run_id: str = "reviewer-v4-english-20260723"

    @property
    def key(self) -> str:
        fold = f"/fold_{self.fold}" if self.fold is not None else ""
        return f"{self.dataset}/{self.model}/{self.protocol}{fold}"


ORIGINAL_SOURCES = {
    ("gender", "qwen"): Path("results/v2/gender/guided_zero_shot_v4/qwen/seed_17"),
    ("gender", "gpt_oss"): Path("results/v2/gender/guided_zero_shot_v4/gpt_oss/seed_17"),
    ("age", "qwen"): Path("results/v2/age/guided_zero_shot_v4__age_opaque/qwen/seed_17"),
    ("age", "gpt_oss"): Path("results/v2/age/guided_zero_shot_v4__age_opaque/gpt_oss/seed_17"),
    ("rosbank", "qwen"): Path("results/v2/rosbank/guided_zero_shot_v4/qwen/seed_17"),
    ("rosbank", "gpt_oss"): Path("results/v2/rosbank/guided_zero_shot_v4/gpt_oss/seed_17"),
}
ORIGINAL_DERIVED = Path("results/v2/derived/reviewer-v6-e5-clustering")
BERKA_RUN_ID = "reviewer-v5-fixed-new-datasets"
BERKA_PROTOCOL = "unittab_70_30_5seed"
BERKA_DERIVED = Path("results/v5/derived/cv_main_e5/berka") / BERKA_PROTOCOL
BERKA_FIDELITY = (
    Path("results/v5/derived/cv_fidelity/berka") / BERKA_PROTOCOL
)
DATAFUSION_RUN_ID = "reviewer-v10-datafusion-public"
DATAFUSION_PROTOCOL = "public_kfold5_seed100"

ARTICLE_VALUES = {
    "gender": {"majority": .575, "direct_llm": .628, "cot": .683, "standard": .774, "handcrafted": .793, "concat": .786},
    "age": {"majority": .260, "direct_llm": .338, "cot": .413, "standard": .642, "handcrafted": .600, "concat": .601},
    "rosbank": {"majority": .553, "direct_llm": .521, "cot": .605, "standard": .743, "handcrafted": .744, "concat": .739},
}
PUBLISHED_BASELINES = [
    ("datafusion_education", "Aggregation", "roc_auc", .793, .013, "MBD five-fold protocol"),
    ("datafusion_education", "CoLES", "roc_auc", .784, .012, "MBD five-fold protocol"),
    ("datafusion_education", "TabBERT", "roc_auc", .762, .014, "MBD five-fold protocol"),
    ("datafusion_education", "TabGPT", "roc_auc", .766, .013, "MBD five-fold protocol"),
    ("datafusion_education", "Supervised RNN", "roc_auc", .712, .015, "MBD five-fold protocol"),
    ("berka", "UniTTab", "positive_f1", .673, .038, "478/204 repeated split"),
    ("berka", "TabBERT", "positive_f1", .620, .024, "478/204 repeated split"),
    ("berka", "LUNA", "positive_f1", .637, .043, "478/204 repeated split"),
    ("berka", "XGBoost", "positive_f1", .608, .079, "478/204 repeated split"),
    ("berka", "CatBoost", "positive_f1", .527, .065, "478/204 repeated split"),
    ("berka", "VAR", "positive_f1", .474, .007, "478/204 repeated split"),
]
MODEL_INDEPENDENT_METHODS = {
    "standard", "handcrafted", "llm_profile", "standard_profile", "all_nonclaim",
}
METRIC_NAMES = (
    "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
    "positive_f1", "mcc", "roc_auc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_count(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for line in file if line.strip())


def safe_git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() or "unknown"


def completion_source_root(marker: Path) -> Path:
    payload = load_json(marker)
    paths = []
    for split in payload.get("split_evidence", {}).values():
        paths.extend(Path(value) for value in split.get("artifacts", {}))
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise ValueError(f"Ambiguous source root in {marker}: {sorted(map(str, parents))}")
    return parents.pop()


def valid_completion(marker: Path, *, dataset: str, model: str, fold: int) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = load_json(marker)
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        payload.get("status") == "completed"
        and payload.get("dataset") == dataset
        and payload.get("model_slug") == model
        and int(payload.get("fold", -1)) == fold
        and payload.get("completion_signature")
    ):
        return False
    for evidence in payload.get("split_evidence", {}).values():
        for raw_path, expected in evidence.get("artifacts", {}).items():
            path = REPO_ROOT / raw_path
            if not path.is_file() or path.stat().st_size != int(expected.get("size", -1)):
                return False
    return True


def canonical_cells() -> tuple[list[Cell], list[dict[str, Any]]]:
    cells: list[Cell] = []
    pending: list[dict[str, Any]] = []
    for (dataset, model), source in ORIGINAL_SOURCES.items():
        required = [source / f"explanations_{split}.jsonl" for split in ("train", "val", "test")]
        required += [source / f"claims_{split}.jsonl" for split in ("train", "val", "test")]
        derived = ORIGINAL_DERIVED / dataset / model / "seed_17"
        if all(path.is_file() for path in required) and (derived / "ml_metrics.json").is_file():
            cells.append(Cell(dataset, model, "v4_updated_split", source, derived))
        else:
            pending.append({"experiment": f"{dataset}/{model}/seed17", "reason": "incomplete canonical artifacts"})

    for model in ("qwen", "gpt_oss"):
        for fold in range(5):
            marker = Path("logs/runs") / BERKA_RUN_ID / "completion" / f"{model}_berka_fold_{fold}.json"
            if valid_completion(marker, dataset="berka", model=model, fold=fold):
                cells.append(Cell(
                    "berka", model, BERKA_PROTOCOL, completion_source_root(marker),
                    BERKA_DERIVED / f"fold_{fold}" / model, fold, BERKA_RUN_ID,
                ))
            else:
                pending.append({"experiment": f"berka/{model}/fold_{fold}", "reason": "missing valid completion marker"})

    df_markers = {
        model: Path("logs/runs") / DATAFUSION_RUN_ID / "completion" / f"{model}_datafusion_education_dataset.json"
        for model in ("qwen", "gpt_oss")
    }
    if all(path.is_file() for path in df_markers.values()):
        for model in ("qwen", "gpt_oss"):
            for fold in range(5):
                marker = Path("logs/runs") / DATAFUSION_RUN_ID / "completion" / f"{model}_datafusion_education_fold_{fold}.json"
                derived = Path("results/v5/derived/cv_main_e5_public/datafusion_education") / DATAFUSION_PROTOCOL / f"fold_{fold}" / model
                if valid_completion(marker, dataset="datafusion_education", model=model, fold=fold) and (derived / "ml_metrics.json").is_file():
                    cells.append(Cell("datafusion_education", model, DATAFUSION_PROTOCOL, completion_source_root(marker), derived, fold, DATAFUSION_RUN_ID))
                else:
                    pending.append({"experiment": f"datafusion/{model}/fold_{fold}/offline", "reason": "API or offline artifact incomplete"})
    else:
        pending.append({"experiment": "datafusion_education/qwen+gpt_oss/5fold", "reason": "dataset-level API completion not reached"})
    if not Path("results/v5/lora/reviewer-v10-lora-all-datasets/original/age/lora/qwen3_8b/metrics.json").is_file():
        pending.append({"experiment": "age/qwen3_8b_lora", "reason": "training is still running"})
    if not list(Path("results/v5/lora/reviewer-v10-lora-all-datasets").glob("datafusion_education/**/metrics.json")):
        pending.append({"experiment": "datafusion_education/qwen3_8b_lora/5fold", "reason": "queued after the current LoRA cell"})
    pending.extend([
        {"experiment": "llm_generation_stability/qwen/seeds_101_947", "reason": "full generation queue is waiting for DataFusion"},
        {"experiment": "llm_generation_stability/gpt_oss/seeds_101_947", "reason": "full generation queue is waiting for DataFusion"},
        {"experiment": "datafusion_education/offline_clusters_ml", "reason": "requires complete claims from both source models"},
        {"experiment": "datafusion_education/grounding_fidelity", "reason": "requires completed API and offline feature artifacts"},
        {"experiment": "cross_model_cluster_matching", "reason": "not yet materialized as a canonical completed artifact"},
        {"experiment": "qwen_32b_lora", "reason": "optional experiment has not been launched"},
    ])
    return cells, pending


class Snapshot:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.files: list[dict[str, Any]] = []
        self.metric_rows: list[dict[str, Any]] = []
        self.cluster_rows: list[dict[str, Any]] = []
        self.claim_samples: list[dict[str, Any]] = []

    def add_file(self, path: Path, *, source: Path | None, role: str, rows: int | None = None) -> None:
        self.files.append({
            "path": str(path.relative_to(self.output)),
            "source": str(source) if source else None,
            "role": role,
            "size": path.stat().st_size,
            "sha256": sha256(path),
            "rows": rows,
        })

    def copy(self, source: Path, relative: Path, *, role: str) -> Path:
        target = self.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows = jsonl_count(target) if target.suffix == ".jsonl" else None
        self.add_file(target, source=source, role=role, rows=rows)
        return target

    def write_text(self, relative: Path, content: str, *, role: str) -> Path:
        target = self.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.add_file(target, source=None, role=role)
        return target

    def write_frame(self, relative: Path, frame: pd.DataFrame, *, role: str) -> Path:
        target = self.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".parquet":
            frame.to_parquet(target, index=False)
        else:
            frame.to_csv(target, index=False)
        self.add_file(target, source=None, role=role, rows=len(frame))
        return target

    def write_jsonl(self, relative: Path, rows: Iterable[dict[str, Any]], *, role: str) -> Path:
        target = self.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        materialized = list(rows)
        with target.open("w", encoding="utf-8") as file:
            for row in materialized:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.add_file(target, source=None, role=role, rows=len(materialized))
        return target


def metric_value(summary: dict[str, Any], name: str) -> tuple[Any, Any]:
    value = summary.get(name, {})
    if isinstance(value, dict):
        return value.get("mean"), value.get("sd")
    return value, None


def metric_row(
    cell: Cell, method: str, classifier: str, summary: dict[str, Any],
    *, test: dict[str, Any] | None = None, n_seeds: int = 1,
) -> dict[str, Any]:
    test = test or {}
    shared = method in MODEL_INDEPENDENT_METHODS
    row = {
        "dataset": cell.dataset, "model": "shared" if shared else cell.model,
        "protocol": cell.protocol, "fold": cell.fold, "method": method,
        "classifier": classifier, "n": test.get("n"), "n_seeds": n_seeds,
    }
    for metric in METRIC_NAMES:
        mean, sd = metric_value(summary, metric)
        row[metric] = mean if mean is not None else test.get(metric)
        row[f"{metric}_sd"] = sd
    return row


def export_metrics(snapshot: Snapshot, cell: Cell) -> None:
    direct = cell.source_root / "llm_metrics_test.json"
    if direct.is_file():
        payload = load_json(direct)
        row = {
            "dataset": cell.dataset, "model": cell.model, "protocol": cell.protocol,
            "fold": cell.fold, "method": "direct_llm", "classifier": "direct_label",
            "n": payload.get("n_scored"), "n_seeds": 1,
        }
        for metric in METRIC_NAMES:
            row[metric] = payload.get(metric)
            row[f"{metric}_sd"] = None
        snapshot.metric_rows.append(row)
        snapshot.copy(direct, Path("metrics/raw/direct_llm") / cell.dataset / cell.model / (f"fold_{cell.fold}" if cell.fold is not None else "seed_17") / "test_metrics.json", role="direct_llm_metrics")

    if not cell.derived_root:
        return
    ml_path = cell.derived_root / "ml_metrics.json"
    if ml_path.is_file():
        payload = load_json(ml_path)
        for method, result in payload.items():
            if not isinstance(result, dict):
                continue
            if method in MODEL_INDEPENDENT_METHODS and cell.model != "qwen":
                continue
            for classifier, details in result.items():
                if classifier in {"artifact_signature", "seeds"} or not isinstance(details, dict):
                    continue
                summary = details.get("summary", {}).get("test", {})
                test = details.get("test", {})
                if not summary and not test:
                    continue
                snapshot.metric_rows.append(metric_row(
                    cell, method, classifier, summary, test=test,
                    n_seeds=len(details.get("runs", {})) or 1,
                ))
        snapshot.copy(ml_path, Path("metrics/raw/ml") / cell.dataset / cell.model / (f"fold_{cell.fold}" if cell.fold is not None else "seed_17") / "ml_metrics.json", role="ml_metrics")
    optional = cell.derived_root / "optional_booster_metrics.json"
    if optional.is_file():
        snapshot.copy(optional, Path("metrics/raw/ml") / cell.dataset / cell.model / f"fold_{cell.fold}" / "optional_booster_metrics.json", role="optional_booster_metrics")
        payload = load_json(optional)
        for method, classifiers in payload.get("feature_sets", {}).items():
            if method in MODEL_INDEPENDENT_METHODS and cell.model != "qwen":
                continue
            if not isinstance(classifiers, dict):
                continue
            for classifier, details in classifiers.items():
                summary = details.get("test_summary", {}) if isinstance(details, dict) else {}
                runs = details.get("runs", {}) if isinstance(details, dict) else {}
                first_run = next(iter(runs.values()), {}) if isinstance(runs, dict) else {}
                if summary:
                    snapshot.metric_rows.append(metric_row(
                        cell, method, classifier, summary, test=first_run,
                        n_seeds=len(runs) or 1,
                    ))


def export_text_artifacts(snapshot: Snapshot, cell: Cell) -> None:
    destination = Path("rationales_and_claims") / cell.dataset / cell.model / cell.protocol
    if cell.fold is not None:
        destination /= f"fold_{cell.fold}"
    for split in ("train", "val", "test"):
        for source_name, target_name, role in (
            (f"explanations_{split}.jsonl", f"cot_{'validation' if split == 'val' else split}.jsonl", "cot_rationales"),
            (f"claims_{split}.jsonl", f"claims_{'validation' if split == 'val' else split}.jsonl", "atomic_claims"),
        ):
            source = cell.source_root / source_name
            if source.is_file():
                snapshot.copy(source, destination / target_name, role=role)
        explanation = cell.source_root / f"explanations_{split}.jsonl"
        if explanation.is_file():
            with explanation.open(encoding="utf-8") as file:
                for index, line in enumerate(file):
                    if index >= 5:
                        break
                    row = json.loads(line)
                    snapshot.claim_samples.append({
                        "dataset": cell.dataset, "model": cell.model,
                        "protocol": cell.protocol, "fold": cell.fold, "split": split,
                        "customer_id": row.get("customer_id"),
                        "prediction": row.get("prediction"), "label": row.get("label"),
                        "explanation": str(row.get("explanation", ""))[:4000],
                    })


def export_clusters(snapshot: Snapshot, cell: Cell) -> None:
    if not cell.derived_root:
        return
    clusters_path = cell.derived_root / "cot_clusters.json"
    if not clusters_path.is_file():
        return
    payload = load_json(clusters_path)
    meta = payload.get("cluster_meta", [])
    rows = []
    representatives = []
    for item in meta:
        label_counts = item.get("assigned_label_counts") or item.get("label_counts") or {}
        total = sum(int(value) for value in label_counts.values()) or 1
        row = {
            "dataset": cell.dataset, "model": cell.model, "protocol": cell.protocol,
            "fold": cell.fold, "cluster_id": item.get("cluster_id"),
            "feature": item.get("feature"), "medoid": item.get("medoid"),
            "unique_claims": item.get("unique_claims"), "occurrences": item.get("occurrences"),
            "unique_clients": item.get("unique_clients"),
            "compactness_mean_distance": item.get("compactness_mean_distance"),
            "assignment_distance_p95": item.get("assignment_distance_p95"),
            "label_counts_json": json.dumps(label_counts, ensure_ascii=False, sort_keys=True),
            "dominant_label_share": max(map(int, label_counts.values()), default=0) / total,
            "embedding_model": payload.get("embedding_model"),
            "selection_mode": payload.get("selection_mode"),
        }
        rows.append(row)
        for rank, claim in enumerate(item.get("examples", [])[:10], 1):
            representatives.append({**{key: row[key] for key in ("dataset", "model", "protocol", "fold", "cluster_id")}, "rank": rank, "claim": claim})
    frame = pd.DataFrame(rows)
    destination = Path("clusters") / cell.dataset / cell.model / (f"fold_{cell.fold}" if cell.fold is not None else "seed_17")
    snapshot.write_frame(destination / "cluster_catalog.csv", frame, role="cluster_catalog")
    snapshot.write_text(destination / "cluster_catalog.json", json.dumps(rows, ensure_ascii=False, indent=2), role="cluster_catalog")
    snapshot.write_frame(destination / "representative_claims.csv", pd.DataFrame(representatives), role="representative_claims")
    snapshot.cluster_rows.extend(rows)

    model_path = cell.derived_root / "cot_cluster_model.npz"
    if model_path.is_file() and rows:
        centroids = np.load(model_path)["centroids"]
        count = min(len(centroids), len(rows))
        if count >= 2:
            coordinates = PCA(n_components=2, random_state=17).fit_transform(centroids[:count])
        else:
            coordinates = np.zeros((count, 2))
        points = frame.iloc[:count].copy()
        points["x"] = coordinates[:, 0]
        points["y"] = coordinates[:, 1]
        points["projection"] = "PCA of frozen train centroids; visualization only"
        snapshot.write_frame(destination / "cluster_map_points.parquet", points, role="cluster_map")
    for source_name, target_name in (
        ("claim_assignments_train.parquet", "claim_assignments_train.parquet"),
        ("claim_assignments_val.parquet", "claim_assignments_validation.parquet"),
        ("claim_assignments_test.parquet", "claim_assignments_test.parquet"),
        ("claim_assignments_outer_train.parquet", "claim_assignments_outer_train.parquet"),
        ("cot_features_train.parquet", "cluster_features_train.parquet"),
        ("cot_features_val.parquet", "cluster_features_validation.parquet"),
        ("cot_features_test.parquet", "cluster_features_test.parquet"),
        ("cot_features_outer_train.parquet", "cluster_features_outer_train.parquet"),
    ):
        source = cell.derived_root / source_name
        if source.is_file():
            snapshot.copy(source, destination / target_name, role="cluster_assignments_or_features")
    for source in (cell.derived_root / "cluster_selection.json", cell.derived_root / "stages/selected_features.json"):
        if source.is_file():
            snapshot.copy(source, destination / source.name, role="cluster_selection")


def export_stability(snapshot: Snapshot, cells: list[Cell]) -> pd.DataFrame:
    rows = []
    seen = set()
    for cell in cells:
        if not cell.derived_root or cell.fold is not None:
            continue
        base = cell.derived_root / "stability"
        for axis, path in (
            ("clustering_seed", base / "cluster_seeds/summary.json"),
            ("granularity", base / "granularity.json"),
        ):
            if not path.is_file() or (cell.dataset, cell.model, axis) in seen:
                continue
            seen.add((cell.dataset, cell.model, axis))
            payload = load_json(path)
            values = payload.get("rows", payload if isinstance(payload, list) else [])
            if isinstance(values, list):
                for value in values:
                    rows.append({"dataset": cell.dataset, "model": cell.model, "axis": axis, **value})
            snapshot.copy(path, Path("metrics/stability/raw") / cell.dataset / cell.model / f"{axis}.json", role="stability_metrics")
    frame = pd.DataFrame(rows)
    berka_summary = BERKA_DERIVED / "stability_summary.json"
    if berka_summary.is_file():
        snapshot.copy(berka_summary, Path("metrics/stability/raw/berka/stability_summary.json"), role="stability_metrics")
        payload = load_json(berka_summary)
        values = payload.get("rows", payload if isinstance(payload, list) else [])
        if isinstance(values, list):
            berka_rows = pd.DataFrame([{"dataset": "berka", "axis": "fold_and_cluster_seed", **value} for value in values])
            frame = pd.concat([frame, berka_rows], ignore_index=True, sort=False)
    snapshot.write_frame(Path("metrics/stability/stability_results.csv"), frame, role="stability_summary")
    return frame


def export_fidelity(snapshot: Snapshot) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, confidence_rows = [], []
    root = Path("results/v2/derived/fidelity-v2")
    for path in sorted(root.glob("**/fidelity_metrics.json")):
        relative = path.relative_to(root)
        if len(relative.parts) < 5:
            continue
        dataset, feature_source = relative.parts[:2]
        payload = load_json(path)
        snapshot.copy(path, Path("metrics/fidelity/raw") / relative, role="fidelity_metrics")
        protocol = payload.get("protocol", {})
        selected = payload.get("surrogate_selection", {}).get("selected_surrogate")
        teacher_seed = protocol.get("teacher_seed")
        surrogate_seed = protocol.get("surrogate_seed")
        for family, splits in payload.get("results", {}).items():
            if not isinstance(splits, dict) or family in {"prior_control", "permutation_control"}:
                continue
            metrics = splits.get("test")
            if not isinstance(metrics, dict):
                continue
            base = {
                "dataset": dataset, "feature_source": feature_source,
                "teacher_seed": teacher_seed, "surrogate_seed": surrogate_seed,
                "surrogate_family": family, "selected_family": family == selected,
            }
            summary_rows.append({**base, **{key: metrics.get(key) for key in (
                "hard_agreement", "probability_mae", "probability_rmse",
                "jensen_shannon_divergence", "agree_and_correct", "agree_and_wrong",
                "teacher_only_correct", "surrogate_only_correct",
            )}})
            for point in metrics.get("confidence_coverage", []):
                confidence_rows.append({**base, **point})

    # Berka uses a five-fold CV protocol rather than the seed-pair layout of
    # the original three datasets.  Preserve each source model independently:
    # combining Qwen and GPT-OSS claim spaces would answer a different
    # question and would hide cross-model variation.
    if (BERKA_FIDELITY / "fidelity_summary.json").is_file():
        for source_name in ("fidelity_summary.json", "fidelity_by_fold.csv"):
            snapshot.copy(
                BERKA_FIDELITY / source_name,
                Path("fidelity/berka") / source_name,
                role="berka_cv_fidelity_summary",
            )
        for fold in range(5):
            fold_root = BERKA_FIDELITY / f"fold_{fold}"
            teacher_selection = fold_root / "teacher/teacher_selection.json"
            if teacher_selection.is_file():
                snapshot.copy(
                    teacher_selection,
                    Path("fidelity/berka") / f"fold_{fold}"
                    / "teacher_selection.json",
                    role="berka_cv_fidelity_teacher",
                )
            for feature_source in ("qwen", "gpt_oss"):
                cell = fold_root / feature_source
                metrics_path = cell / "fidelity_metrics.json"
                if not metrics_path.is_file():
                    continue
                payload = load_json(metrics_path)
                selected = payload.get("surrogate_selection", {}).get(
                    "selected_surrogate"
                )
                protocol = payload.get("protocol", {})
                base = {
                    "dataset": "berka",
                    "feature_source": feature_source,
                    "protocol": BERKA_PROTOCOL,
                    "fold": fold,
                    "teacher_seed": protocol.get("teacher_seed", 17),
                    "surrogate_seed": protocol.get("surrogate_seed", 17),
                }
                for family, splits in payload.get("results", {}).items():
                    if (
                        not isinstance(splits, dict)
                        or family in {"prior_control", "permutation_control"}
                    ):
                        continue
                    metrics = splits.get("test")
                    if not isinstance(metrics, dict):
                        continue
                    row_base = {
                        **base,
                        "surrogate_family": family,
                        "selected_family": family == selected,
                    }
                    summary_rows.append({
                        **row_base,
                        **{key: metrics.get(key) for key in (
                            "hard_agreement", "probability_mae",
                            "probability_rmse",
                            "jensen_shannon_divergence",
                            "agree_and_correct", "agree_and_wrong",
                            "teacher_only_correct", "surrogate_only_correct",
                        )},
                    })
                    for point in metrics.get("confidence_coverage", []):
                        confidence_rows.append({**row_base, **point})
                destination = (
                    Path("fidelity/berka") / f"fold_{fold}" / feature_source
                )
                for source_name in (
                    "fidelity_metrics.json",
                    "surrogate_selection.json",
                    "surrogate_predictions.csv",
                    "cluster_occlusion.jsonl",
                    "tree_decision_paths.jsonl",
                    "shallow_tree.txt",
                    "fidelity_stage.json",
                ):
                    source = cell / source_name
                    if source.is_file():
                        snapshot.copy(
                            source,
                            destination / source_name,
                            role="berka_cv_fidelity_artifact",
                        )
    summary = pd.DataFrame(summary_rows)
    confidence = pd.DataFrame(confidence_rows)
    snapshot.write_frame(Path("fidelity/fidelity_summary.csv"), summary, role="fidelity_summary")
    snapshot.write_frame(Path("fidelity/high_confidence_agreement.csv"), confidence, role="fidelity_confidence")
    snapshot.write_frame(Path("fidelity/confidence_coverage_curves.csv"), confidence, role="fidelity_confidence")
    selected_confidence = confidence[confidence["selected_family"] == True] if not confidence.empty else confidence
    if not selected_confidence.empty:
        selected_confidence = selected_confidence.copy()
        selected_confidence["threshold"] = pd.to_numeric(
            selected_confidence["threshold"], errors="coerce"
        ).round(2)
        keys = ["dataset", "feature_source", "threshold"]
        high_summary = selected_confidence.groupby(keys, as_index=False).agg(
            agreement_mean=("agreement", "mean"), agreement_sd=("agreement", "std"),
            coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"),
            n_seed_pairs=("agreement", "size"),
        )
        snapshot.write_frame(Path("fidelity/high_confidence_summary.csv"), high_summary, role="fidelity_confidence_summary")
    for name in ("cluster_occlusion.jsonl", "tree_decision_paths.jsonl"):
        sources = sorted(
            path for path in root.glob(f"**/{name}")
            if "teacher_seed_17" in path.parts and "surrogate_seed_17" in path.parts
        )
        if sources:
            rows = []
            for source in sources:
                with source.open(encoding="utf-8") as file:
                    for line in file:
                        if line.strip():
                            row = json.loads(line)
                            row["source_path"] = str(source)
                            rows.append(row)
            frame = pd.DataFrame(rows)
            target_name = "cluster_occlusion.parquet" if name.startswith("cluster") else "tree_decision_paths.parquet"
            snapshot.write_frame(Path("fidelity") / target_name, frame, role=name.removesuffix(".jsonl"))
            if name.startswith("cluster"):
                snapshot.write_frame(Path("fidelity/cluster_occlusion.csv"), frame, role="cluster_occlusion")
            else:
                snapshot.write_jsonl(Path("fidelity/tree_decision_paths.jsonl"), rows, role="tree_decision_paths")
    if not summary.empty:
        quadrants = summary[[
            "dataset", "feature_source", "teacher_seed", "surrogate_seed",
            "surrogate_family", "selected_family", "agree_and_correct",
            "agree_and_wrong", "teacher_only_correct", "surrogate_only_correct",
        ]]
        snapshot.write_frame(Path("fidelity/outcome_quadrants.csv"), quadrants, role="fidelity_outcomes")
    return summary, confidence


def copy_analysis_reports(snapshot: Snapshot) -> pd.DataFrame:
    mappings = [
        (Path("reports/reviewer-v9-grounding-official-ready/detailed"), Path("grounding_review")),
        (Path("reports/reviewer-v10-stereotype-audit/stereotype_audit"), Path("legacy/stereotype_audit_v4")),
        (Path("reports/reviewer-v10-stereotype-audit-legacy/stereotype_audit"), Path("legacy/stereotype_audit_legacy")),
        (Path("reports/reviewer-v10-stereotype-audit-comparison"), Path("legacy/legacy_vs_v4_stereotype_comparison")),
    ]
    for source_root, destination in mappings:
        if not source_root.is_dir():
            continue
        for source in source_root.iterdir():
            if source.is_file() and source.suffix in {".md", ".json", ".csv", ".html"}:
                snapshot.copy(source, destination / source.name, role="reviewer_analysis")
    judgments = Path("reports/reviewer-v9-grounding-official-ready/detailed/judgments.csv")
    if judgments.is_file():
        frame = pd.read_csv(judgments)
        verdict = frame.get("verdict", pd.Series(index=frame.index, dtype=str)).astype(str)
        partial = frame[verdict == "partially_supported"]
        snapshot.write_frame(Path("grounding_review/partially_supported_claims.csv"), partial, role="grounding_review_subset")
        unsupported = frame[verdict == "unsupported"]
        snapshot.write_frame(Path("grounding_review/unsupported_claims.csv"), unsupported, role="grounding_review_subset")
        grouped = (frame.groupby(["dataset", "judge_name", "verdict"], dropna=False).size()
                   .rename("judgments").reset_index())
        grouped["share_within_dataset_judge"] = grouped["judgments"] / grouped.groupby(
            ["dataset", "judge_name"]
        )["judgments"].transform("sum")
        snapshot.write_frame(Path("grounding_review/grounding_summary.csv"), grouped, role="grounding_summary")
        disagreements = Path("reports/reviewer-v9-grounding-official-ready/detailed/disagreements.csv")
        if disagreements.is_file():
            snapshot.copy(disagreements, Path("grounding_review/judge_disagreements.csv"), role="grounding_review_subset")
        manual = Path("reports/reviewer-v9-grounding-official-ready/detailed/manual_validation_blinded.json")
        if manual.is_file():
            values = load_json(manual)
            if isinstance(values, dict):
                values = values.get("items", values.get("rows", []))
            if isinstance(values, list):
                snapshot.write_jsonl(Path("grounding_review/manual_validation_sample.jsonl"), values, role="manual_validation")
        return grouped
    return pd.DataFrame()


def export_lora(snapshot: Snapshot) -> pd.DataFrame:
    rows = []
    paths = [
        *Path("results/v5/lora/reviewer-v10-lora-all-datasets/original").glob("*/lora/qwen3_8b/metrics.json"),
        *Path("results/v5/lora/reviewer-v9-lora-qwen3-8b/berka").glob("**/lora/qwen3_8b/metrics.json"),
        *Path("results/v5/lora/reviewer-v10-lora-all-datasets").glob("datafusion_education/**/lora/qwen3_8b/metrics.json"),
    ]
    for path in sorted(set(paths)):
        payload = load_json(path)
        dataset = "berka" if "berka" in path.parts else ("datafusion_education" if "datafusion_education" in path.parts else path.parts[-4])
        fold = next((int(part.split("_")[1]) for part in path.parts if part.startswith("fold_")), None)
        rows.append({
            "dataset": dataset, "model": "qwen3_8b_lora", "fold": fold,
            **{key: payload.get(f"test_{key}") for key in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "roc_auc")},
            "positive_f1": payload.get("test_f1_positive"),
        })
        snapshot.copy(path, Path("metrics/lora_metrics/raw") / dataset / (f"fold_{fold}" if fold is not None else "single_split") / "metrics.json", role="lora_metrics")
    frame = pd.DataFrame(rows)
    snapshot.write_frame(Path("metrics/lora_metrics/lora_results.csv"), frame, role="lora_summary")
    return frame


def aggregate_fold_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "protocol", "method", "classifier"]
    numeric = [column for column in ("accuracy", "balanced_accuracy", "f1_macro", "positive_f1", "mcc", "roc_auc") if column in metrics]
    rows = []
    for identity, group in metrics.groupby(keys, dropna=False):
        row = dict(zip(keys, identity))
        row["n_cells"] = len(group)
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = values.mean() if len(values) else None
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else group[f"{metric}_sd"].dropna().mean() if f"{metric}_sd" in group else None
        rows.append(row)
    return pd.DataFrame(rows)


def article_deltas(aggregated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    method_map = {"cot": "cot", "standard": "standard", "handcrafted": "handcrafted", "concat": "concat", "direct_llm": "direct_llm"}
    for dataset, submitted in ARTICLE_VALUES.items():
        for article_method, current_method in method_map.items():
            models = ("shared",) if current_method in MODEL_INDEPENDENT_METHODS else ("qwen", "gpt_oss")
            for model in models:
                candidates = aggregated[(aggregated.dataset == dataset) & (aggregated.model == model) & (aggregated.method == current_method)]
                if current_method != "direct_llm":
                    candidates = candidates[candidates.classifier == "xgboost"]
                if candidates.empty:
                    continue
                current = candidates.iloc[0].get("accuracy")
                if pd.isna(current):
                    continue
                rows.append({
                    "dataset": dataset, "model": model, "method": article_method,
                    "submitted_accuracy": submitted[article_method], "current_accuracy": current,
                    "delta_current_minus_submitted": current - submitted[article_method],
                    "note": "Submitted and current results use different splits; this is not a paired test.",
                })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3, limit: int | None = None) -> str:
    if frame.empty:
        return "_Нет завершённых данных._"
    view = frame[columns].copy()
    if limit:
        view = view.head(limit)
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    return view.to_markdown(index=False)


def save_figure(snapshot: Snapshot, relative: Path, figure: Any, *, role: str) -> None:
    target = snapshot.output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight", dpi=220)
    plt.close(figure)
    snapshot.add_file(target, source=None, role=role)


def export_figures(
    snapshot: Snapshot,
    aggregated: pd.DataFrame,
    stability: pd.DataFrame,
    confidence: pd.DataFrame,
) -> None:
    palette = {"qwen": "#3567d6", "gpt_oss": "#159a8c", "shared": "#d9822b"}
    for dataset, group in aggregated.groupby("dataset"):
        selected = group[group["classifier"].isin(["xgboost", "catboost", "direct_label"])].copy()
        selected = selected.sort_values("accuracy", ascending=False).head(16)
        if selected.empty:
            continue
        labels = [f"{row.model}\n{row.method}/{row.classifier}" for row in selected.itertuples()]
        values = selected["accuracy"].fillna(selected["balanced_accuracy"]).fillna(0).to_numpy()
        fig, ax = plt.subplots(figsize=(12, 6))
        positions = np.arange(len(selected))
        colors = [palette.get(str(model), "#7f8aa3") for model in selected["model"]]
        ax.barh(positions, values, color=colors)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Accuracy")
        ax.set_title(f"{dataset}: завершённые методы")
        ax.grid(axis="x", alpha=.18)
        for y, value in zip(positions, values):
            ax.text(min(value + .008, .96), y, f"{value:.3f}", va="center", fontsize=8)
        save_figure(snapshot, Path("figures/performance") / f"{dataset}_accuracy.svg", fig, role="performance_figure")

    if not stability.empty and {"ari", "nmi"}.issubset(stability.columns):
        view = stability.dropna(subset=["ari", "nmi"])
        if not view.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            for (dataset, model), group in view.groupby(["dataset", "model"]):
                ax.scatter(group["ari"], group["nmi"], label=f"{dataset}/{model}", alpha=.8, s=55)
            ax.set(xlabel="ARI", ylabel="NMI", title="Устойчивость cluster assignments")
            ax.set_xlim(-.05, 1.05); ax.set_ylim(-.05, 1.05); ax.grid(alpha=.2)
            ax.legend(fontsize=7, ncol=2)
            save_figure(snapshot, Path("figures/stability/clustering_ari_nmi.svg"), fig, role="stability_figure")

    selected_confidence = confidence[confidence.get("selected_family", False) == True] if not confidence.empty else confidence
    if not selected_confidence.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for (dataset, source), group in selected_confidence.groupby(["dataset", "feature_source"]):
            mean = group.groupby("threshold", as_index=False)[["coverage", "agreement"]].mean()
            label = f"{dataset}/{source}"
            axes[0].plot(mean["threshold"], mean["agreement"], marker="o", label=label)
            axes[1].plot(mean["threshold"], mean["coverage"], marker="o", label=label)
        axes[0].set(title="Fidelity уверенных teacher-ответов", ylabel="Agreement")
        axes[1].set(title="Confidence–coverage", ylabel="Coverage")
        for ax in axes:
            ax.set_xlabel("Teacher confidence threshold"); ax.set_ylim(0, 1.02); ax.grid(alpha=.2)
        axes[1].legend(fontsize=6, ncol=2)
        save_figure(snapshot, Path("figures/fidelity/confidence_coverage.svg"), fig, role="fidelity_figure")

    for path in snapshot.output.glob("clusters/**/cluster_map_points.parquet"):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        sizes = 18 + 150 * pd.to_numeric(frame.get("unique_clients", 1), errors="coerce").fillna(1) / max(pd.to_numeric(frame.get("unique_clients", 1), errors="coerce").max(), 1)
        color = pd.to_numeric(frame.get("dominant_label_share", .5), errors="coerce").fillna(.5)
        scatter = ax.scatter(frame["x"], frame["y"], s=sizes, c=color, cmap="viridis", alpha=.75, edgecolor="white", linewidth=.35)
        top = frame.nlargest(min(5, len(frame)), "unique_clients")
        for index, row in enumerate(top.itertuples(), 1):
            ax.annotate(str(index), (row.x, row.y), fontsize=9, fontweight="bold")
        title = "/".join(path.relative_to(snapshot.output / "clusters").parts[:-1])
        ax.set_title(f"Claim clusters · {title}"); ax.set_xlabel("PCA-1"); ax.set_ylabel("PCA-2")
        ax.grid(alpha=.12); fig.colorbar(scatter, ax=ax, label="Dominant train-label share")
        relative = Path("figures/clusters") / ("__".join(path.relative_to(snapshot.output / "clusters").parts[:-1]) + ".svg")
        save_figure(snapshot, relative, fig, role="cluster_figure")

    grounding = snapshot.output / "grounding_review/judgments.csv"
    if grounding.is_file():
        frame = pd.read_csv(grounding)
        if "verdict" in frame:
            counts = frame["verdict"].value_counts(normalize=True).sort_values()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.barh(counts.index, counts.values, color="#3567d6")
            ax.set(xlabel="Доля judgments", title="Grounding verdict distribution", xlim=(0, 1))
            ax.grid(axis="x", alpha=.2)
            save_figure(snapshot, Path("figures/grounding/verdict_distribution.svg"), fig, role="grounding_figure")


def build_experiment_doc(
    aggregated: pd.DataFrame,
    deltas: pd.DataFrame,
    stability: pd.DataFrame,
    fidelity: pd.DataFrame,
    confidence: pd.DataFrame,
    grounding: pd.DataFrame,
    lora: pd.DataFrame,
    pending: list[dict[str, Any]],
) -> str:
    best = aggregated.sort_values(["dataset", "accuracy"], ascending=[True, False]).groupby("dataset", as_index=False).head(12)
    stability_summary = stability[stability.get("axis", pd.Series(dtype=str)) == "clustering_seed"].copy() if not stability.empty else stability
    confidence_selected = confidence[(confidence.get("selected_family", False) == True) & (confidence.get("threshold", 0) >= .8)].copy() if not confidence.empty else confidence
    confidence_summary = pd.DataFrame()
    if not confidence_selected.empty:
        confidence_selected["threshold"] = pd.to_numeric(
            confidence_selected["threshold"], errors="coerce"
        ).round(2)
        confidence_summary = confidence_selected.groupby(
            ["dataset", "feature_source", "threshold"], as_index=False
        ).agg(coverage=("coverage", "mean"), agreement=("agreement", "mean"), n_seed_pairs=("agreement", "size"))
        confidence_summary = confidence_summary[confidence_summary["threshold"].isin([.8, .9, .95])]
    berka_fidelity = pd.DataFrame()
    if not fidelity.empty and "dataset" in fidelity:
        selected_berka = fidelity[
            fidelity["dataset"].eq("berka")
            & fidelity.get("selected_family", False).eq(True)
        ].copy()
        if not selected_berka.empty:
            berka_fidelity = selected_berka.groupby(
                "feature_source", as_index=False
            ).agg(
                n_folds=("fold", "nunique"),
                hard_agreement_mean=("hard_agreement", "mean"),
                hard_agreement_sd=("hard_agreement", "std"),
                probability_mae_mean=("probability_mae", "mean"),
                jensen_shannon_mean=(
                    "jensen_shannon_divergence", "mean"
                ),
                agree_and_correct_mean=("agree_and_correct", "mean"),
                agree_and_wrong_mean=("agree_and_wrong", "mean"),
            )
    pending_lines = "\n".join(f"- `{row['experiment']}` — {row['reason']}." for row in pending)
    return f"""# Полное описание экспериментов и текущих результатов

Снимок построен {time.strftime('%Y-%m-%d %H:%M:%S %Z')}. В численные таблицы входят только завершённые cells с проверенными артефактами. Частичные API-файлы не интерпретируются как финальный результат.

## 1. Экспериментальный протокол

Для каждого клиента LLM получает train-derived class summaries и его агрегированный транзакционный профиль, генерирует свободное evidence-bounded объяснение и один итоговый label. Из объяснения отдельным deterministic-temperature запросом извлекаются атомарные claims. Claims нормализуются, одинаковые тексты embedding-вычисляются один раз, а occurrence-to-client mapping сохраняется.

Semantic clusters строятся только по train claims. Validation используется для выбора granularity, encoding и downstream hyperparameters; test claims назначаются к замороженным train centroids. Labels validation/test не участвуют в построении feature space.

Основные признаки: `standard` — универсальные транзакционные агрегаты; `handcrafted` — task-specific признаки; `llm_profile` — численные признаки из robust LLM profile; `cot` — cluster features; `concat` — handcrafted + CoT; `standard_cot` — standard + CoT; `all_features` — объединение всех доступных non-claim и cluster features с train-only selection.

## 2. Общее качество

Таблица ниже содержит mean по ML seeds или folds. Для single-run Direct LLM SD отсутствует.

{markdown_table(best, ['dataset','model','method','classifier','n_cells','accuracy','balanced_accuracy','f1_macro','positive_f1','roc_auc'])}

Главный общий паттерн: direct LLM обычно слабее табличных классификаторов, тогда как claims превращают объяснения в повторно используемое семантическое пространство. На обновлённой E5-кластеризации CoT заметно усилился относительно ранних результатов, а `standard_cot`/`all_features` позволяют отфильтровать шум и использовать clusters только там, где они дают дополнительный сигнал.

## 3. Сравнение с submitted article

{markdown_table(deltas, ['dataset','model','method','submitted_accuracy','current_accuracy','delta_current_minus_submitted'])}

Эти дельты показывают совместный эффект обновлённого split, английских evidence-bounded prompts, нового embedding/clustering pipeline и повторной ML-оценки. Их нельзя интерпретировать как чистую однофакторную абляцию. Классические агрегаты меняются меньше, чем LLM-derived representation; Rosbank остаётся наиболее благоприятной основной задачей для семантических features.

## 4. Stability

Clustering seeds `17, 101, 947` сравниваются через ARI/NMI на общем train claim space и через variation validation performance. Granularity (`k=100,200,400,800`) репортится отдельно и не смешивается с seed stability.

{markdown_table(stability_summary, [c for c in ['dataset','model','seed','candidate','n_clusters_after_coverage','validation_balanced_accuracy','ari','nmi','joint_assignment_coverage'] if c in stability_summary.columns], limit=30)}

ARI измеряет совпадение пар claims с поправкой на случайность; NMI — долю общей информации между двумя partitions. Значения около 1 означают почти идентичную структуру, около 0 — отсутствие устойчивого соответствия. Downstream variation необходимо читать вместе с ARI/NMI: разные partitions могут сохранять близкую predictive utility.

Полные LLM-generation seeds `101` и `947` ещё не завершены; имеющиеся checkpoints не включены в итоговые числа.

## 5. Grounding

Автоматический grounding проведён для 840 claims (Age, Gender, Rosbank, Berka), по два независимых judge на claim: 1,680 judgments. Judge получает exact hashed client profile, train reference summary, field semantics и claim, но не true label и не source-model identity. Вердикты: supported, partially supported, unsupported и not verifiable. Расхождения судей не превращаются в искусственное majority; подготовлено 127 blinded manual-validation items.

Подробные breakdown, unsupported cases и disagreements находятся в `grounding_review/`. Основная зона риска — не прямые наблюдения, а train-relative comparisons и более высокоуровневые behavioral interpretations. Поэтому в статье следует отдельно сообщать результаты по claim type и judge.

{markdown_table(grounding, ['dataset','judge_name','verdict','judgments','share_within_dataset_judge'], limit=80)}

## 6. Аудит стереотипов

На одинаковых клиентах сравнивались legacy prompts с явными age/gender heuristics и v4 evidence-bounded prompts. Для Age строгий DeepSeek-judge показал снижение broad problematic rate на 21.7–38.3 п.п.; для Gender Gemini показал снижение на 10.0–13.3 п.п. Самое устойчивое улучшение — сокращение external group generalizations, categorical personal attributes и top-k absence errors. Согласие judges низкое, поэтому обязательна ручная проверка priority subset.

## 7. Fidelity и объяснимость уверенных teacher-ответов

Teacher выбирается по validation среди сильных non-claim classifiers. Claim-based logistic regression, XGBoost и shallow tree обучаются воспроизводить teacher probabilities; это surrogate fidelity, а не causal explanation.

{markdown_table(fidelity[fidelity.get('selected_family', False) == True] if not fidelity.empty else fidelity, ['dataset','feature_source','teacher_seed','surrogate_seed','surrogate_family','hard_agreement','probability_mae','jensen_shannon_divergence','agree_and_correct','agree_and_wrong'], limit=30)}

### Berka: five-fold fidelity

Для Berka teacher независимо выбирался внутри каждого fold по inner validation, после чего frozen teacher и validation-selected claim surrogate оценивались на соответствующем outer test. Qwen и GPT-OSS рассматриваются как два отдельных пространства признаков; `union` и `semantic_union` здесь намеренно не используются. Таблица показывает mean ± sample SD по пяти folds.

{markdown_table(berka_fidelity, ['feature_source','n_folds','hard_agreement_mean','hard_agreement_sd','probability_mae_mean','jensen_shannon_mean','agree_and_correct_mean','agree_and_wrong_mean'])}

Оба пространства восстанавливают hard prediction сильного non-claim teacher примерно в 93–94% случаев. Qwen имеет немного более высокое agreement и меньшие probability MAE/JS divergence. При teacher confidence ≥0.90 agreement возрастает примерно до 98%, сохраняя около 87% клиентов; при confidence ≥0.95 agreement составляет около 98–99% при coverage около 80%. Это сильный результат surrogate fidelity, но не доказательство причинной верности каждого отдельного claim.

Для уверенных teacher predictions agreement вычисляется при thresholds 0.50–0.95. Это непосредственно отвечает на вопрос, могут ли clusters объяснять решения сильной модели там, где сама модель уверена.

{markdown_table(confidence_summary, ['dataset','feature_source','threshold','coverage','agreement','n_seed_pairs'], limit=80)}

Важна confidence–coverage trade-off: при росте threshold agreement обычно повышается, но объяснение покрывает меньшую часть клиентов. `agree_and_correct` и `agree_and_wrong` разделяют верное воспроизведение решения и совместную ошибку teacher/surrogate. Occlusion и decision paths лежат в `fidelity/`.

## 8. LoRA

{markdown_table(lora, [c for c in ['dataset','model','fold','accuracy','balanced_accuracy','f1_macro','positive_f1','roc_auc'] if c in lora.columns])}

Qwen3-8B LoRA для Berka, Rosbank и Gender завершён. Age и DataFusion добавятся после появления `metrics.json`. Qwen-32B пока не запускался.

## 9. Что ещё будет

{pending_lines}

После завершения очередей следует повторить `build_paper_data.py --execute`: completed artifacts добавятся без ручного выбора и без изменения source results.
"""


def build_architecture_doc() -> str:
    return """# Архитектура проекта и экспериментального pipeline

## Data flow

```text
Raw transaction splits
  → dataset-aware robust client profiles
  → train-only class summaries and optional factual demonstrations
  → adaptive Qwen / GPT-OSS generation
  → direct label + behavioral rationale
  → atomic claims extraction
  → normalization / text deduplication / E5 embeddings
  → train-only semantic clustering
  → frozen validation/test assignment
  → binary/count/normalized-count cluster features
  → train-only supervised feature selection
  → ML, shallow trees and feature concatenations
  → stability, grounding, stereotype audit and surrogate fidelity
  → paper_data and publication reports
```

## Data boundary

Label-conditioned summaries, few-shot demonstrations, semantic cluster formation и supervised feature selection используют только train. Validation применяется для выбора prompt variant, cluster granularity и ML hyperparameters. Test используется один раз после фиксации configuration. Изменение test labels не должно менять summaries, demonstrations, centroids или feature construction.

## Generation и recovery

Qwen и GPT-OSS имеют независимые adaptive schedulers. Стартовый atomic window — 64. Transport/rate-limit failures откатывают только transport window и переводят scheduler на 10, затем при необходимости на 1. Content errors (`MissingFinalAnswer`, parser failure, empty claims) не откатывают валидные ответы: проблемные request keys отправляются в repair queue после основного прохода. Checkpoints и `pending_batch.json` обеспечивают crash-safe resume.

На уровне model queue действует глобальный process lease: одновременно разрешена одна очередь Qwen и одна GPT-OSS. Разные модели работают параллельно, datasets внутри модели — последовательно.

## Claims и clustering

Каждый claim хранит stable identity, original/normalized text, customer ID, source explanation hash и extractor signature. Дедупликация embeddings выполняется по normalized text, но client occurrences не теряются. Финальный embedding model — `intfloat/multilingual-e5-large`. UMAP/PCA используется только для визуализации, не как stability metric.

Semantic clustering label-agnostic. Label association считается после образования clusters и только на train. Validation/test claims назначаются к frozen centroids. Основной encoding — binary presence, поскольку он не превращает длину rationale в скрытый признак; count encodings остаются абляциями.

## Feature families

- `standard`: общие activity, amount, temporal и category aggregates.
- `handcrafted`: dataset-specific признаки, спроектированные без LLM.
- `llm_profile`: robust mean/median/IQR/P5–P95 indicators, доступные prompt.
- `cot`: только semantic cluster features.
- `concat`: handcrafted + cot.
- `standard_cot`: standard + cot.
- `all_nonclaim`: объединение численных non-claim feature families.
- `all_features`: all_nonclaim + semantic clusters с train-only selection.

## Дополнительные эксперименты

- Clustering stability: seeds 17/101/947, ARI/NMI и downstream variation.
- Granularity: k=100/200/400/800 и coverage thresholds.
- Generation stability: полные LLM reruns с seeds 101/947.
- Grounding: exact client evidence, два независимых judges и blinded human subset.
- Stereotype audit: legacy versus evidence-bounded v4 на одинаковых клиентах.
- Fidelity: validation-selected teacher, probability surrogate, confidence–coverage, occlusion и exact shallow-tree paths.

## Главные entrypoints

- `run_pipeline.py`: базовый DAG stats → prompts → explanations → evaluation → claims.
- `scripts/run_cv_llm_queue.py`: fold-based Berka/DataFusion API queue.
- `scripts/run_v4_offline_pipeline.py`: embeddings, clusters, features и seeded ML.
- `scripts/run_fidelity_suite.py`: teacher selection и surrogate fidelity.
- `scripts/run_cv_fidelity.py`: fold-pure Berka fidelity без объединения LLM feature spaces.
- `scripts/run_grounding_suite.py`: sampling и judge evaluation.
- `scripts/build_paper_data.py`: completed-only аналитический snapshot.
"""


def data_dictionary() -> str:
    return """# Словарь файлов и полей

## Метрики

`metrics/main_results_long.csv` содержит одну строку на dataset/model/protocol/fold/method/classifier. `main_results_wide.csv` агрегирует folds и seeds. Суффикс `_sd` означает sample standard deviation. Пустое значение означает, что метрика неприменима или был только один run.

## CoT и claims

`cot_*.jsonl` — полный сохранённый response на клиента: customer identity, explanation, parsed prediction, label, signatures и execution metadata. `claims_*.jsonl` — список атомарных behavioral claims, извлечённых из соответствующего explanation.

## Clusters

`cluster_catalog` содержит stable cluster ID, medoid, representative texts, coverage, compactness и post-hoc train label counts. `claim_assignments` связывает claim occurrence с frozen cluster и distance. `cluster_features` — client-level sparse/wide representation для ML. Координаты карты являются только визуальной проекцией centroids.

## Grounding

`unsupported_cases.csv`, `disagreements.csv` и `manual_validation.html` сохраняют exact evidence и judge rationale для ручной проверки. Verdict judges не является human gold label.

## Fidelity

`hard_agreement` — доля совпадающих hard predictions teacher и surrogate. MAE/RMSE измеряют расхождение probabilities; Jensen–Shannon — симметричное distribution divergence. `coverage` — доля клиентов выше teacher-confidence threshold; `agreement` — fidelity внутри этой подвыборки.

Для Berka `fidelity/berka/fidelity_by_fold.csv` и `fidelity_summary.json` содержат отдельные результаты Qwen и GPT-OSS по пяти outer-test folds. Каталоги `fold_N/{qwen,gpt_oss}` содержат predictions, occlusion, confidence–coverage и exact tree paths; пространства двух генераторов не объединяются.

## Provenance

`manifest.json` содержит source path, SHA-256, размер и row count каждого файла. `checksums.sha256` позволяет проверить переносимость snapshot.
"""


def build_html(
    aggregated: pd.DataFrame,
    deltas: pd.DataFrame,
    stability: pd.DataFrame,
    fidelity: pd.DataFrame,
    confidence: pd.DataFrame,
    clusters: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    pending: list[dict[str, Any]],
) -> str:
    def records(frame: pd.DataFrame, limit: int = 5000) -> str:
        return frame.head(limit).where(pd.notna(frame), None).to_json(orient="records", force_ascii=False)
    data = {
        "metrics": json.loads(records(aggregated)), "deltas": json.loads(records(deltas)),
        "stability": json.loads(records(stability)), "fidelity": json.loads(records(fidelity)),
        "confidence": json.loads(records(confidence)), "clusters": clusters[:5000],
        "samples": samples[:500], "pending": pending,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>Paper data · аналитический отчёт</title>
<style>:root{{--navy:#14213d;--blue:#3567d6;--teal:#159a8c;--bg:#f2f5fa}}*{{box-sizing:border-box}}body{{margin:0;font:14px system-ui;color:var(--navy);background:var(--bg)}}header{{padding:32px 5%;color:white;background:linear-gradient(125deg,var(--navy),var(--blue))}}h1{{margin:0 0 8px}}nav{{display:flex;gap:7px;position:sticky;top:0;padding:12px 4%;background:white;overflow:auto;box-shadow:0 2px 12px #0001}}button{{border:0;border-radius:8px;padding:9px 13px;cursor:pointer}}button.active{{background:var(--blue);color:white}}main{{max-width:1500px;margin:auto;padding:22px}}section{{display:none}}section.active{{display:block}}.card{{background:white;border-radius:15px;padding:19px;margin-bottom:18px;box-shadow:0 7px 24px #14213d10;overflow:auto}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.kpi{{background:white;border-radius:13px;padding:16px}}.kpi b{{font-size:25px;display:block}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #e5e8ef;text-align:left;max-width:420px}}input,select{{padding:8px;border:1px solid #ccd4e2;border-radius:7px;margin:4px}}pre{{white-space:pre-wrap;max-height:380px;overflow:auto;background:#f5f7fb;padding:12px}}.bar{{height:11px;background:#dce5f8;border-radius:8px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--teal)}}@media(max-width:800px){{.kpis{{grid-template-columns:1fr 1fr}}}}</style></head><body>
<header><h1>Paper data · текущие экспериментальные результаты</h1><p>Completed-only snapshot · {time.strftime('%Y-%m-%d %H:%M:%S %Z')} · интерактивные таблицы, clusters, grounding и fidelity.</p></header>
<nav id='nav'></nav><main id='main'></main><script>const D={payload};
const tabs=['Обзор','Метрики','Legacy deltas','Stability','Clusters','CoT browser','Fidelity','High-confidence','Pending','Файлы'];
function esc(x){{return String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}};
function table(rows,cols){{if(!rows.length)return '<p>Нет завершённых данных.</p>';cols=cols||Object.keys(rows[0]);return `<table><thead><tr>${{cols.map(c=>`<th>${{esc(c)}}</th>`).join('')}}</tr></thead><tbody>${{rows.map(r=>`<tr>${{cols.map(c=>`<td>${{typeof r[c]==='number'?r[c].toFixed(4):esc(r[c])}}</td>`).join('')}}</tr>`).join('')}}</tbody></table>`}}
function show(i){{document.querySelectorAll('section').forEach((x,j)=>x.classList.toggle('active',i===j));document.querySelectorAll('nav button').forEach((x,j)=>x.classList.toggle('active',i===j))}}
document.getElementById('nav').innerHTML=tabs.map((x,i)=>`<button onclick='show(${{i}})'>${{x}}</button>`).join('');
const best=[...D.metrics].sort((a,b)=>(b.accuracy||0)-(a.accuracy||0)).slice(0,30);
const sections=[];
sections.push(`<section><div class='kpis'><div class='kpi'><b>${{D.metrics.length}}</b>metric cells</div><div class='kpi'><b>${{D.clusters.length}}</b>clusters</div><div class='kpi'><b>${{D.samples.length}}</b>CoT samples</div><div class='kpi'><b>${{D.pending.length}}</b>pending</div></div><div class='card'><h2>Лучшие завершённые cells</h2>${{table(best,['dataset','model','method','classifier','accuracy','balanced_accuracy','roc_auc'])}}</div></section>`);
sections.push(`<section><div class='card'><h2>Все агрегированные метрики</h2>${{table(D.metrics)}}</div></section>`);
sections.push(`<section><div class='card'><h2>Submitted article → current</h2>${{table(D.deltas)}}</div></section>`);
sections.push(`<section><div class='card'><h2>Clustering stability</h2>${{table(D.stability)}}</div></section>`);
sections.push(`<section><div class='card'><h2>Cluster explorer</h2><input id='cf' placeholder='dataset/model/text' oninput='clusters()'><div id='ct'></div></div></section>`);
sections.push(`<section><div class='card'><h2>CoT browser</h2><input id='rf' placeholder='dataset/model/text' oninput='rationales()'><div id='rt'></div></div></section>`);
sections.push(`<section><div class='card'><h2>Surrogate fidelity</h2>${{table(D.fidelity)}}</div></section>`);
sections.push(`<section><div class='card'><h2>Teacher confidence → coverage/agreement</h2>${{table(D.confidence)}}</div></section>`);
sections.push(`<section><div class='card'><h2>Незавершённые эксперименты</h2>${{table(D.pending)}}</div></section>`);
sections.push(`<section><div class='card'><h2>Навигация</h2><p><a href='../README.md'>README</a> · <a href='../EXPERIMENTS_AND_RESULTS.md'>результаты</a> · <a href='../PIPELINE_ARCHITECTURE.md'>архитектура</a> · <a href='../manifest.json'>manifest</a></p></div></section>`);
document.getElementById('main').innerHTML=sections.join('');
function clusters(){{let q=(document.getElementById('cf')?.value||'').toLowerCase(),r=D.clusters.filter(x=>JSON.stringify(x).toLowerCase().includes(q)).slice(0,300);document.getElementById('ct').innerHTML=table(r,['dataset','model','fold','cluster_id','medoid','unique_clients','compactness_mean_distance','dominant_label_share'])}}
function rationales(){{let q=(document.getElementById('rf')?.value||'').toLowerCase(),r=D.samples.filter(x=>JSON.stringify(x).toLowerCase().includes(q)).slice(0,80);document.getElementById('rt').innerHTML=r.map(x=>`<article><h3>${{esc(x.dataset)}} · ${{esc(x.model)}} · ${{esc(x.split)}} · ${{esc(x.customer_id)}}</h3><pre>${{esc(x.explanation)}}</pre></article>`).join('')}}
show(0);clusters();rationales();</script></body></html>"""


def build_readme(cells: list[Cell], pending: list[dict[str, Any]]) -> str:
    datasets = sorted({cell.dataset for cell in cells})
    return f"""# Paper data

Переносимый completed-only snapshot для написания long paper, reviewer response и построения графиков.

## Быстрый вход

- [`EXPERIMENTS_AND_RESULTS.md`](EXPERIMENTS_AND_RESULTS.md) — все готовые эксперименты, метрики и интерпретация.
- [`PIPELINE_ARCHITECTURE.md`](PIPELINE_ARCHITECTURE.md) — архитектура, data boundary и смысл feature families.
- [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) — схемы файлов и определение метрик.
- [`STATUS_AND_PENDING.md`](STATUS_AND_PENDING.md) — что включено и что ещё считается.
- [`REVIEWS.md`](REVIEWS.md) — полный исходный текст трёх reviewer reports.
- [`report/index.html`](report/index.html) — интерактивный локальный отчёт.
- [`manifest.json`](manifest.json) и [`checksums.sha256`](checksums.sha256) — provenance.

Включённые datasets: {', '.join(datasets)}. Канонических model/fold cells: {len(cells)}. Pending entries: {len(pending)}.

Полные CoT и claims находятся в `rationales_and_claims/`; embeddings и checkpoints намеренно не дублируются. Исходные результаты остаются в `results/` и не изменяются.

## Обновление

```bash
python scripts/build_paper_data.py --output paper_data --execute
```

Без `--execute` команда только печатает план.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("paper_data"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cells, pending = canonical_cells()
    plan = {
        "mode": "execute" if args.execute else "dry-run", "output": str(args.output),
        "completed_cells": [cell.key for cell in cells], "pending": pending,
        "policy": "full CoT/claims; curated metrics/clusters; no embeddings/checkpoints",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return

    output = args.output.resolve()
    if output == REPO_ROOT.resolve() or REPO_ROOT.resolve() not in output.parents:
        raise ValueError("paper_data output must be inside the repository and not its root")
    output.mkdir(parents=True, exist_ok=True)
    for generated in ("metrics", "rationales_and_claims", "clusters", "grounding_review", "fidelity", "legacy", "figures", "report", "source_material"):
        path = output / generated
        if path.exists():
            shutil.rmtree(path)
    snapshot = Snapshot(output)

    for cell in cells:
        export_text_artifacts(snapshot, cell)
        export_metrics(snapshot, cell)
        export_clusters(snapshot, cell)
    snapshot.write_jsonl(Path("rationales_and_claims/cot_analysis_sample.jsonl"), snapshot.claim_samples, role="cot_analysis_sample")
    snapshot.write_frame(Path("rationales_and_claims/cot_analysis_sample.csv"), pd.DataFrame(snapshot.claim_samples), role="cot_analysis_sample")
    metrics = pd.DataFrame(snapshot.metric_rows)
    aggregated = aggregate_fold_metrics(metrics)
    deltas = article_deltas(aggregated)
    snapshot.write_frame(Path("metrics/main_results_long.csv"), metrics, role="main_metrics")
    snapshot.write_frame(Path("metrics/main_results_wide.csv"), aggregated, role="main_metrics")
    snapshot.write_frame(Path("metrics/legacy_article_deltas.csv"), deltas, role="legacy_comparison")
    baselines = pd.DataFrame(PUBLISHED_BASELINES, columns=["dataset", "baseline", "metric", "mean", "sd", "protocol_note"])
    snapshot.write_frame(Path("metrics/published_baselines.csv"), baselines, role="published_baselines")

    stability = export_stability(snapshot, cells)
    fidelity, confidence = export_fidelity(snapshot)
    lora = export_lora(snapshot)
    grounding = copy_analysis_reports(snapshot)
    for source, target in (
        (Path("ARR_May_2026___Transactions-1.pdf"), Path("source_material/submitted_article.pdf")),
        (Path("submission12583_reviews.md"), Path("source_material/reviewer_comments.md")),
        (Path("submission12583_reviews.md"), Path("REVIEWS.md")),
    ):
        if source.is_file():
            snapshot.copy(source, target, role="source_material")
    export_figures(snapshot, aggregated, stability, confidence)

    docs = {
        "README.md": build_readme(cells, pending),
        "EXPERIMENTS_AND_RESULTS.md": build_experiment_doc(aggregated, deltas, stability, fidelity, confidence, grounding, lora, pending),
        "PIPELINE_ARCHITECTURE.md": build_architecture_doc(),
        "DATA_DICTIONARY.md": data_dictionary(),
        "STATUS_AND_PENDING.md": "# Статус и ожидаемые результаты\n\n" + "\n".join(f"- `{row['experiment']}`: {row['reason']}." for row in pending),
    }
    for name, content in docs.items():
        snapshot.write_text(Path(name), content, role="documentation")
    snapshot.write_text(Path("report/index.html"), build_html(aggregated, deltas, stability, fidelity, confidence, snapshot.cluster_rows, snapshot.claim_samples, pending), role="interactive_report")
    snapshot.write_text(Path(".gitignore"), "rationales_and_claims/\nclusters/**/*.parquet\nfidelity/*.parquet\n", role="git_policy")

    manifest = {
        "snapshot_version": 1, "created_at": time.time(), "git_revision": safe_git_revision(),
        "completed_cells": [cell.__dict__ | {"source_root": str(cell.source_root), "derived_root": str(cell.derived_root) if cell.derived_root else None} for cell in cells],
        "pending": pending, "files": sorted(snapshot.files, key=lambda row: row["path"]),
        "exclusions": ["embedding matrices", "model checkpoints", "caches", "duplicated full ml_predictions"],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({"built": str(output), "files": len(checksums), "size_bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file())}, indent=2))


if __name__ == "__main__":
    main()

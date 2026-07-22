"""Select a non-claim teacher using validation metrics only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.artifacts import atomic_write_json, files_fingerprint


def read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    return pd.read_csv(path)


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    for prefix in ("teacher_prob_", "probability_"):
        columns = [column for column in frame if column.startswith(prefix)]
        if columns:
            try:
                return sorted(columns, key=lambda column: int(column[len(prefix):]))
            except ValueError as exc:
                raise ValueError(f"Non-numeric probability class suffix: {columns}") from exc
    raise ValueError(
        "Teacher predictions require teacher_prob_<class> or probability_<class> columns"
    )


def _candidate_parts(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(spec) == {"train", "val", "test"}:
        return spec, {}
    paths = spec.get("paths", {})
    filters = spec.get("filters", {})
    if set(paths) != {"train", "val", "test"}:
        raise ValueError("Each teacher must define train/val/test paths")
    if not isinstance(filters, dict):
        raise ValueError("Teacher filters must be a mapping")
    return paths, filters


def evaluate_candidate(
    path: Path,
    *,
    filters: dict[str, Any] | None = None,
    split: str = "val",
) -> dict[str, Any]:
    frame = read_predictions(path)
    for column, value in (filters or {}).items():
        if column not in frame:
            raise ValueError(f"Missing teacher filter column: {column}")
        frame = frame[frame[column] == value]
    if "split" in frame:
        frame = frame[frame["split"] == split]
    if frame.empty:
        raise ValueError(f"No teacher predictions remain for split={split}: {path}")
    if frame["customer_id"].duplicated().any():
        raise ValueError(f"Duplicate teacher customer IDs for split={split}: {path}")
    columns = _probability_columns(frame)
    probabilities = frame[columns].to_numpy(float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4)
    ):
        raise ValueError(f"Invalid teacher probability distributions: {path}")
    truth = frame["label"].to_numpy(int)
    prediction = probabilities.argmax(axis=1)
    return {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.candidates_config.read_text(encoding="utf-8"))
    teachers = config.get("teachers", {})
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "selection_split": "validation",
        "candidates": teachers,
        "primary_metric": "balanced_accuracy",
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    if not teachers:
        raise ValueError("No teachers configured")

    metrics = {}
    normalized = {}
    for name, spec in teachers.items():
        paths, filters = _candidate_parts(spec)
        missing = [str(path) for path in paths.values() if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing teacher files for {name}: {missing}")
        normalized[name] = {"paths": paths, "filters": filters}
        metrics[name] = evaluate_candidate(
            Path(paths["val"]),
            filters=filters,
            split="val",
        )
    selected = sorted(
        teachers,
        key=lambda name: (
            -metrics[name]["balanced_accuracy"],
            -metrics[name]["macro_f1"],
            name,
        ),
    )[0]
    selected_paths = {
        key: str(Path(value))
        for key, value in normalized[selected]["paths"].items()
    }
    selected_filters = normalized[selected]["filters"]
    payload = {
        "selection_split": "val",
        "primary_metric": "balanced_accuracy",
        "selected_teacher": selected,
        "selected": {
            "paths": selected_paths,
            "filters": selected_filters,
            "file_hashes": files_fingerprint(selected_paths.values()),
        },
        "candidate_metrics": metrics,
    }
    atomic_write_json(args.output, payload)
    print(f"Selected teacher {selected} -> {args.output}")


if __name__ == "__main__":
    main()

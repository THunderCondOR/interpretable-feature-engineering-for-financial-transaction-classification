from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data.loader import add_features, load_dataset
from src.models.ml_baseline import run_ml_baseline


# ---------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "label_names" in config.get("dataset", {}):
        config["dataset"]["label_names"] = {
            str(k): v for k, v in config["dataset"]["label_names"].items()
        }

    return config


def get_out_dir(config: dict) -> Path:
    out = Path(config["output"]["base_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def split_path(config: dict, key: str, split: str) -> Path:
    out = get_out_dir(config)
    base = Path(config["output"][key])
    suffix = base.suffix or ".jsonl"
    return out / f"{base.stem}_{split}{suffix}"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    out = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_score is not None:
        try:
            out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except Exception as exc:
            out["roc_auc_error"] = str(exc)

    return out


# ---------------------------------------------------------------------
# 1. CoT feature sanity check
# ---------------------------------------------------------------------

def check_cot_features(config: dict, splits: list[str]) -> dict:
    out = get_out_dir(config)
    report = {}

    for split in splits:
        path = out / f"cot_features_{split}.parquet"

        if not path.exists():
            report[split] = {
                "exists": False,
                "path": str(path),
            }
            continue

        df = pd.read_parquet(path)
        feature_cols = [c for c in df.columns if c.startswith("cot_cluster_")]

        if feature_cols:
            row_mass = df[feature_cols].sum(axis=1)
            total_mass = float(df[feature_cols].sum().sum())
            nonzero_rows = int((row_mass > 0).sum())
            zero_rows = int((row_mass == 0).sum())
        else:
            total_mass = 0.0
            nonzero_rows = 0
            zero_rows = int(len(df))

        report[split] = {
            "exists": True,
            "path": str(path),
            "shape": list(df.shape),
            "clients": int(df["customer_id"].nunique()) if "customer_id" in df.columns else None,
            "labels": {
                str(k): int(v)
                for k, v in df["label"].value_counts().sort_index().to_dict().items()
            } if "label" in df.columns else {},
            "n_cot_features": int(len(feature_cols)),
            "nonzero_rows": nonzero_rows,
            "zero_rows": zero_rows,
            "total_feature_mass": total_mass,
        }

    meta_path = out / "cot_cluster_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        report["clusters"] = {
            "path": str(meta_path),
            "n_clusters": len(meta),
            "first_cluster": meta[0] if meta else None,
        }
    else:
        report["clusters"] = {
            "path": str(meta_path),
            "exists": False,
        }

    path = out / "cot_feature_report.json"
    write_json(path, report)
    print(f"[check] CoT feature report saved -> {path}")

    return report


# ---------------------------------------------------------------------
# 2. LLM baseline, single sample per client
# ---------------------------------------------------------------------

def evaluate_llm_single_sample(config: dict, splits: list[str]) -> dict:
    """
    Direct LLM baseline.

    Protocol:
      one LLM answer per client.

    If explanations file contains multiple samples per client, this function
    uses sample_id == 0, not majority vote. This keeps LLM baseline separate
    from majority/self-consistency experiments.
    """
    out = get_out_dir(config)
    results = {}

    for split in splits:
        path = split_path(config, "explanations", split)

        if not path.exists():
            results[split] = {
                "exists": False,
                "path": str(path),
            }
            continue

        rows = read_jsonl(path)

        by_client = defaultdict(list)
        for r in rows:
            by_client[int(r["customer_id"])].append(r)

        pred_rows = []
        y_true, y_pred, y_score = [], [], []

        n_total = len(by_client)
        n_scored = 0
        n_unscored = 0

        for cid, group in by_client.items():
            group = sorted(group, key=lambda x: int(x.get("sample_id", 0)))

            chosen = None
            for r in group:
                if int(r.get("sample_id", 0)) == 0:
                    chosen = r
                    break
            if chosen is None and group:
                chosen = group[0]

            label = int(chosen.get("label", -1)) if chosen else -1
            pred = chosen.get("predicted") if chosen else None
            raw = chosen.get("predicted_raw") if chosen else None
            error = chosen.get("error") if chosen else None

            if pred is not None:
                pred = int(pred)

            if pred in (0, 1):
                churn_score = float(pred)
                n_scored += 1
            else:
                pred = -1
                churn_score = None
                n_unscored += 1

            pred_rows.append({
                "customer_id": cid,
                "label": label,
                "prediction": pred,
                "churn_score": churn_score,
                "predicted_raw": raw,
                "sample_id_used": chosen.get("sample_id") if chosen else None,
                "n_available_samples": len(group),
                "error": error,
            })

            if label >= 0 and pred >= 0 and churn_score is not None:
                y_true.append(label)
                y_pred.append(pred)
                y_score.append(churn_score)

        pred_path = out / f"llm_predictions_{split}.csv"
        pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

        if y_true:
            metrics = compute_metrics(y_true, y_pred, y_score)
            metrics.update({
                "n_total_clients": int(n_total),
                "n_scored_clients": int(n_scored),
                "n_unscored_clients": int(n_unscored),
                "coverage": round(float(n_scored / n_total), 4) if n_total else 0.0,
                "llm_eval_protocol": "single_sample_per_customer",
                "roc_auc_score_type": "hard_single_sample_churn_prediction",
                "predictions_path": str(pred_path),
            })
        else:
            metrics = {
                "n_total_clients": int(n_total),
                "n_scored_clients": int(n_scored),
                "n_unscored_clients": int(n_unscored),
                "coverage": round(float(n_scored / n_total), 4) if n_total else 0.0,
                "llm_eval_protocol": "single_sample_per_customer",
                "metrics_skipped": "labels or valid predictions unavailable",
                "predictions_path": str(pred_path),
            }

        results[split] = metrics
        print(f"[llm] {split}: {metrics}")

    path = out / "metrics_llm.json"
    write_json(path, results)
    print(f"[llm] metrics saved -> {path}")

    return results


# ---------------------------------------------------------------------
# 3. Legacy-style clean ML baseline
# ---------------------------------------------------------------------

def safe_name(x: object, max_len: int = 80) -> str:
    s = str(x).replace("ё", "е")
    s = re.sub(r"[^0-9a-zA-Zа-яА-Я_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "unknown"


def build_legacy_features(
    df: pd.DataFrame,
    category_col: str = "mcc_code_desc",
    fitted_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Legacy-style features by analogy with gender_ml_baseline.ipynb:
      - one row per customer
      - global amount aggregates
      - category-level pivot features
    """
    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(int)

    labels = (
        df.drop_duplicates("customer_id")[["customer_id", "label"]]
        .set_index("customer_id")
        .sort_index()
    )

    base = df.groupby("customer_id")["amount"].agg(
        amount_count="count",
        amount_sum="sum",
        amount_mean="mean",
        amount_min="min",
        amount_max="max",
        amount_median="median",
        amount_std="std",
    )

    if "tr_datetime" in df.columns:
        dt = pd.to_datetime(df["tr_datetime"], errors="coerce")
        df["_month"] = dt.dt.to_period("M").astype(str)
        df["_day"] = dt.dt.date

        active_days = df.groupby("customer_id")["_day"].nunique().rename("active_days")
        active_months = df.groupby("customer_id")["_month"].nunique().rename("active_months")
        base = base.join(active_days, how="left").join(active_months, how="left")

        base["txn_per_day"] = base["amount_count"] / base["active_days"].replace(0, np.nan)
        base["txn_per_month"] = base["amount_count"] / base["active_months"].replace(0, np.nan)

    pivots = []
    for suffix, agg_func in {
        "count": "count",
        "sum": "sum",
        "mean": "mean",
        "min": "min",
        "max": "max",
        "median": "median",
    }.items():
        p = (
            df.groupby(["customer_id", category_col])["amount"]
            .agg(agg_func)
            .unstack(category_col)
            .fillna(0)
        )
        p.columns = [f"{category_col}_{safe_name(c)}_{suffix}" for c in p.columns]
        pivots.append(p)

    features = pd.concat([base] + pivots, axis=1).fillna(0)
    features = labels.join(features, how="left").fillna(0).reset_index()

    if fitted_columns is not None:
        for col in fitted_columns:
            if col not in features.columns:
                features[col] = 0.0
        features = features[["customer_id", "label"] + fitted_columns]

    return features


def split_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = df[df["label"] >= 0].copy()
    return df[feature_cols].values, df["label"].astype(int).values


def evaluate_model(model, X, y) -> dict:
    pred = model.predict(X)
    score = None

    if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
        score = model.predict_proba(X)[:, 1]

    return compute_metrics(y, pred, score)


def run_legacy_raw_ml_baseline(config: dict) -> dict:
    out = get_out_dir(config)

    print("[raw-ml] loading splits")
    train_df = add_features(load_dataset(config, "train"))
    val_df = add_features(load_dataset(config, "val"))
    test_df = add_features(load_dataset(config, "test"))

    print("[raw-ml] building legacy aggregate features")
    train_feat = build_legacy_features(train_df)
    feature_cols = [c for c in train_feat.columns if c not in {"customer_id", "label"}]

    val_feat = build_legacy_features(val_df, fitted_columns=feature_cols)
    test_feat = build_legacy_features(test_df, fitted_columns=feature_cols)

    X_train, y_train = split_xy(train_feat, feature_cols)
    X_val, y_val = split_xy(val_feat, feature_cols)
    X_test, y_test = split_xy(test_feat, feature_cols)

    print(f"[raw-ml] train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=8,
        verbosity=0,
    )

    dt = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
    )

    print("[raw-ml] training XGBoost")
    xgb.fit(X_train, y_train)

    print("[raw-ml] training DecisionTree")
    dt.fit(X_train, y_train)

    results = {
        "description": (
            "Legacy-style clean ML baseline: client-level amount aggregates "
            "and category pivots, no LLM, no CoT, no embeddings."
        ),
        "n_features": int(len(feature_cols)),
        "models": {
            "xgboost_legacy_raw_aggregates": {
                "val": evaluate_model(xgb, X_val, y_val),
                "test": evaluate_model(xgb, X_test, y_test),
            },
            "decision_tree_legacy_raw_aggregates": {
                "val": evaluate_model(dt, X_val, y_val),
                "test": evaluate_model(dt, X_test, y_test),
            },
        },
    }

    path = out / "legacy_raw_ml_baseline_metrics.json"
    write_json(path, results)
    print(f"[raw-ml] metrics saved -> {path}")

    return results


# ---------------------------------------------------------------------
# 4. Summary table
# ---------------------------------------------------------------------

def _get_lora_metric(metrics: dict, split: str, name: str):
    """
    Robustly extract LoRA metric from several possible formats:
      test_accuracy / val_accuracy
      eval_accuracy
      accuracy
    """
    candidates = [
        f"{split}_{name}",
        f"{split}_{name.replace('roc_auc', 'roc_auc')}",
        f"eval_{name}" if split == "val" else None,
        name if split in {"test", "val"} else None,
    ]

    for key in candidates:
        if key and key in metrics:
            return metrics[key]

    return None


def _extract_lora_split_metrics(metrics: dict, split: str) -> dict | None:
    """
    Convert LoRA metrics.json into the same flat schema as other baselines.
    Returns None if the split is unavailable.
    """
    accuracy = _get_lora_metric(metrics, split, "accuracy")

    # If there is no split-specific accuracy, do not invent the row.
    if accuracy is None:
        return None

    out = {
        "accuracy": accuracy,
        "roc_auc": _get_lora_metric(metrics, split, "roc_auc"),
        "balanced_accuracy": _get_lora_metric(metrics, split, "balanced_accuracy"),
        "f1_macro": _get_lora_metric(metrics, split, "f1_macro"),
        "f1_weighted": _get_lora_metric(metrics, split, "f1_weighted"),
        "mcc": _get_lora_metric(metrics, split, "mcc"),
        "n": _get_lora_metric(metrics, split, "n"),
        "coverage": 1.0,
    }

    # Some Trainer outputs only eval_accuracy and eval_loss.
    # Then we still keep accuracy but leave the rest empty.
    return out


def collect_lora_rows(config: dict, rows: list[dict]) -> None:
    """
    Add LoRA runs from:
      results/rosbank_1.5B/metrics.json
      results/rosbank_7B/metrics.json
      results/rosbank_14B/metrics.json

    Also supports results/rosbank_lora_summary.json if present.
    """
    out = get_out_dir(config)
    results_root = out.parent
    dataset_name = config["dataset"]["name"]

    # Prefer per-run folders.
    lora_dirs = sorted(results_root.glob(f"{dataset_name}_*B"))

    for run_dir in lora_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        size = run_dir.name.replace(f"{dataset_name}_", "")
        method = f"lora_{size}"

        for split in ["val", "test"]:
            split_metrics = _extract_lora_split_metrics(metrics, split)
            if split_metrics is not None:
                add_metric_row(
                    rows=rows,
                    method=method,
                    split=split,
                    metrics=split_metrics,
                    family="lora",
                )

    # Also support combined summary file, if it exists.
    summary_path = results_root / f"{dataset_name}_lora_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        for run_name, metrics in summary.items():
            if not isinstance(metrics, dict):
                continue
            if metrics.get("status") not in {None, "ok"}:
                continue

            size = run_name.replace(f"{dataset_name}_", "")
            method = f"lora_{size}"

            for split in ["val", "test"]:
                split_metrics = _extract_lora_split_metrics(metrics, split)
                if split_metrics is not None:
                    add_metric_row(
                        rows=rows,
                        method=method,
                        split=split,
                        metrics=split_metrics,
                        family="lora",
                    )

def add_metric_row(rows: list[dict], method: str, split: str, metrics: dict, family: str) -> None:
    n = metrics.get("n") or metrics.get("n_scored_clients")

    coverage = metrics.get("coverage")
    if coverage is None:
        if metrics.get("accuracy") is not None:
            coverage = 1.0

    rows.append({
        "family": family,
        "method": method,
        "split": split,
        "accuracy": metrics.get("accuracy"),
        "roc_auc": metrics.get("roc_auc"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
        "mcc": metrics.get("mcc"),
        "n": n,
        "coverage": coverage,
    })


def collect_summary(config: dict) -> pd.DataFrame:
    out = get_out_dir(config)
    rows = []

    # Majority
    path = out / "metrics_majority.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        for split, metrics in obj.get("splits", {}).items():
            add_metric_row(rows, "majority", split, metrics, "baseline")

    # LLM
    path = out / "metrics_llm.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        for split, metrics in obj.items():
            add_metric_row(rows, "gpt_oss_120b_single_sample", split, metrics, "llm")

    # LoRA
    collect_lora_rows(config, rows)

    # Legacy raw ML
    path = out / "legacy_raw_ml_baseline_metrics.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        for model_name, res in obj.get("models", {}).items():
            for split in ["val", "test"]:
                if split in res:
                    add_metric_row(rows, model_name, split, res[split], "raw_ml")

    # Main ML: cot / handcrafted / concat
    path = out / "ml_metrics.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        for feature_set, models in obj.items():
            for model_name, res in models.items():
                method = f"{model_name}_{feature_set}"
                for split in ["val", "test"]:
                    if split in res:
                        add_metric_row(rows, method, split, res[split], "ml")

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(
            subset=["family", "method", "split"],
            keep="first",
        )
        df = df.sort_values(
            by=["split", "accuracy"],
            ascending=[True, False],
            na_position="last",
        )

    csv_path = out / "rosbank_results_summary.csv"
    md_path = out / "rosbank_results_summary.md"

    df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        if df.empty:
            f.write("No metrics found.\n")
        else:
            f.write(df.to_markdown(index=False))

    print(f"[summary] CSV saved -> {csv_path}")
    print(f"[summary] Markdown saved -> {md_path}")

    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rosbank.yaml")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-llm-eval", action="store_true")
    parser.add_argument("--skip-raw-ml", action="store_true")
    parser.add_argument("--skip-cot-ml", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.optuna_trials is not None:
        config.setdefault("optuna", {})["n_trials"] = args.optuna_trials

    if not args.skip_check:
        check_cot_features(config, splits)

    if not args.skip_llm_eval:
        evaluate_llm_single_sample(config, splits)

    if not args.skip_raw_ml:
        run_legacy_raw_ml_baseline(config)

    if not args.skip_cot_ml:
        print("[ml] running CoT / handcrafted / concat ML baseline")
        run_ml_baseline(config)

    collect_summary(config)


if __name__ == "__main__":
    main()
"""Classical ML baselines for prepared transaction datasets.

Supported feature sets:
- handcrafted: deterministic transaction aggregates;
- cot: train-fitted CoT cluster features;
- concat: handcrafted + CoT features merged by customer_id and label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from src.data.loader import add_features, load_dataset

optuna.logging.set_verbosity(optuna.logging.WARNING)

EXPERIMENTS = {"handcrafted", "cot", "concat"}


def safe_name(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_").replace(",", "")[:100]


def share(part: float, whole: float) -> float:
    return float(part) / float(whole) if whole else 0.0


def build_handcrafted_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if config["dataset"]["name"] == "rosbank":
        return build_rosbank_handcrafted_features(df)
    return build_generic_handcrafted_features(df)


def build_generic_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for customer_id, client in df.groupby("customer_id", sort=False):
        label = int(client["label"].iloc[0])
        n_txn = len(client)
        n_days = max(client["tr_datetime"].dt.date.nunique(), 1) if client["tr_datetime"].notna().any() else 1
        pos = client.loc[client["amount"] > 0, "amount"]
        neg = client.loc[client["amount"] < 0, "amount"]
        row = {
            "customer_id": int(customer_id),
            "label": label,
            "n_txn": n_txn,
            "active_days": n_days,
            "txn_per_day": n_txn / n_days,
            "total_income": float(pos.sum()) if len(pos) else 0.0,
            "total_expense": float(neg.sum()) if len(neg) else 0.0,
            "avg_income": float(pos.mean()) if len(pos) else 0.0,
            "avg_expense": float(neg.mean()) if len(neg) else 0.0,
            "share_income": share(len(pos), n_txn),
            "share_expense": share(len(neg), n_txn),
        }
        if "period_of_day" in client.columns:
            for period in ["утро", "день", "вечер", "ночь"]:
                row[f"share_{period}"] = float((client["period_of_day"] == period).mean())
        if "is_weekend" in client.columns:
            row["share_weekend"] = float(client["is_weekend"].mean())
        for category, count in client.groupby("mcc_code_desc")["amount"].count().to_dict().items():
            row[f"cnt_{safe_name(category)}"] = int(count)
        records.append(row)
    return pd.DataFrame(records).fillna(0)


def build_rosbank_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for customer_id, client in df.groupby("customer_id", sort=False):
        label = int(client["label"].iloc[0])
        n_txn = len(client)
        n_days = max(client["tr_datetime"].dt.date.nunique(), 1) if client["tr_datetime"].notna().any() else 1
        n_months = max(client["tr_datetime"].dt.to_period("M").nunique(), 1) if client["tr_datetime"].notna().any() else 1
        row = {
            "customer_id": int(customer_id),
            "label": label,
            "n_txn": n_txn,
            "active_days": n_days,
            "active_months": n_months,
            "txn_per_day": n_txn / n_days,
            "txn_per_month": n_txn / n_months,
            "total_amount": float(client["amount"].sum()),
            "avg_amount": float(client["amount"].mean()) if n_txn else 0.0,
            "median_amount": float(client["amount"].median()) if n_txn else 0.0,
            "max_amount": float(client["amount"].max()) if n_txn else 0.0,
            "std_amount": float(client["amount"].std()) if n_txn > 1 else 0.0,
            "n_unique_mcc": int(client["mcc_code_desc"].nunique()),
            "mcc_diversity": client["mcc_code_desc"].nunique() / max(n_txn, 1),
        }
        if "trx_cat_ru" in client.columns:
            trx = client["trx_cat_ru"].value_counts()
            atm_total = sum(value for key, value in trx.items() if "снятие" in str(key).lower())
            row.update(
                {
                    "share_pos": share(trx.get("оплата картой", 0), n_txn),
                    "share_atm": share(atm_total, n_txn),
                    "share_deposit": share(trx.get("пополнение счета", 0), n_txn),
                    "share_c2c_out": share(trx.get("перевод на карту", 0), n_txn),
                    "share_c2c_in": share(trx.get("входящий перевод с карты", 0), n_txn),
                }
            )
        if "currency_name" in client.columns:
            row["n_currencies"] = int(client["currency_name"].nunique())
            row["share_rub"] = float((client["currency_name"] == "Рубль").mean())
            row["has_foreign_curr"] = int(client["currency_name"].nunique() > 1)
        if "is_weekend" in client.columns:
            row["share_weekend"] = float(client["is_weekend"].mean())
        if "period_of_day" in client.columns:
            for period in ["утро", "день", "вечер", "ночь"]:
                row[f"share_{period}"] = float((client["period_of_day"] == period).mean())
        if client["tr_datetime"].notna().any() and n_txn >= 4:
            ordered = client.sort_values("tr_datetime")
            start, end = ordered["tr_datetime"].min(), ordered["tr_datetime"].max()
            duration = (end - start).total_seconds()
            if duration > 0:
                mid = start + pd.Timedelta(seconds=duration / 2)
                q1 = start + pd.Timedelta(seconds=duration * 0.25)
                q3 = end - pd.Timedelta(seconds=duration * 0.25)
                first = int((ordered["tr_datetime"] <= mid).sum())
                second = int((ordered["tr_datetime"] > mid).sum())
                early = int((ordered["tr_datetime"] <= q1).sum())
                recent = int((ordered["tr_datetime"] >= q3).sum())
                row["second_to_first_txn_ratio"] = second / max(first, 1)
                row["recent_to_early_txn_ratio"] = recent / max(early, 1)
        category_count = client.groupby("mcc_code_desc")["amount"].count().to_dict()
        category_amount = client.groupby("mcc_code_desc")["amount"].sum().to_dict()
        for category, count in category_count.items():
            name = safe_name(category)
            row[f"cnt_{name}"] = int(count)
            row[f"sum_{name}"] = float(category_amount.get(category, 0.0))
        records.append(row)
    return pd.DataFrame(records).fillna(0)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in {"customer_id", "label"}]


def align_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.reindex(columns=["customer_id", "label"] + columns, fill_value=0)


def split_xy(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return df[columns].values, df["label"].astype(int).values


def load_cot_features(out_dir: Path, split: str) -> pd.DataFrame:
    path = out_dir / f"cot_features_{split}.parquet"
    return pd.read_parquet(path)


def merge_feature_frames(cot: pd.DataFrame, handcrafted: pd.DataFrame) -> pd.DataFrame:
    return cot.merge(handcrafted, on=["customer_id", "label"], how="inner", suffixes=("", "_hc"))


def evaluate(model, x: np.ndarray, y: np.ndarray) -> dict:
    preds = model.predict(x)
    result = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, preds)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, model.predict_proba(x)[:, 1]))
    return result


def primary_score(metrics: dict, config: dict) -> float:
    metric = config["dataset"].get("metric", "accuracy")
    return float(metrics.get(metric, metrics["accuracy"]))


def xgb_objective(config: dict) -> str:
    return "binary:logistic" if int(config["dataset"]["num_labels"]) == 2 else "multi:softprob"


def tune_xgboost(x_train, y_train, x_val, y_val, config: dict, n_trials: int) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
        model = XGBClassifier(
            **params,
            objective=xgb_objective(config),
            random_state=42,
            n_jobs=4,
            verbosity=0,
            eval_metric="logloss",
        )
        model.fit(x_train, y_train)
        return primary_score(evaluate(model, x_val, y_val), config)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_decision_tree(x_train, y_train, x_val, y_val, config: dict, n_trials: int) -> dict:
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }
        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(x_train, y_train)
        return primary_score(evaluate(model, x_val, y_val), config)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def build_feature_sets(config: dict, experiments: list[str]) -> dict[str, dict]:
    requested = set(experiments)
    unknown = requested - EXPERIMENTS
    if unknown:
        raise ValueError(f"Unknown ML experiments: {sorted(unknown)}")

    out_dir = Path(config["output"]["base_dir"])
    train_df = add_features(load_dataset(config, "train"))
    val_df = add_features(load_dataset(config, "val"))
    test_df = add_features(load_dataset(config, "test"))

    feature_sets = {}
    if "handcrafted" in requested or "concat" in requested:
        hc_train_raw = build_handcrafted_features(train_df, config)
        hc_val_raw = build_handcrafted_features(val_df, config)
        hc_test_raw = build_handcrafted_features(test_df, config)
        hc_cols = feature_columns(hc_train_raw)
        hc_train = align_features(hc_train_raw, hc_cols)
        hc_val = align_features(hc_val_raw, hc_cols)
        hc_test = align_features(hc_test_raw, hc_cols)
        if "handcrafted" in requested:
            feature_sets["handcrafted"] = {"train": hc_train, "val": hc_val, "test": hc_test, "columns": hc_cols}

    if "cot" in requested or "concat" in requested:
        cot_train_raw = load_cot_features(out_dir, "train")
        cot_val_raw = load_cot_features(out_dir, "val")
        cot_test_raw = load_cot_features(out_dir, "test")
        cot_cols = feature_columns(cot_train_raw)
        cot_train = align_features(cot_train_raw, cot_cols)
        cot_val = align_features(cot_val_raw, cot_cols)
        cot_test = align_features(cot_test_raw, cot_cols)
        if "cot" in requested:
            feature_sets["cot"] = {"train": cot_train, "val": cot_val, "test": cot_test, "columns": cot_cols}

    if "concat" in requested:
        concat_train = merge_feature_frames(cot_train, hc_train)
        concat_val = merge_feature_frames(cot_val, hc_val)
        concat_test = merge_feature_frames(cot_test, hc_test)
        concat_cols = [column for column in concat_train.columns if column not in {"customer_id", "label"}]
        feature_sets["concat"] = {"train": concat_train, "val": concat_val, "test": concat_test, "columns": concat_cols}

    return feature_sets


def run_ml_baseline(config: dict, experiments: list[str] | None = None) -> None:
    experiments = experiments or ["handcrafted", "cot", "concat"]
    n_trials = int(config.get("optuna", {}).get("n_trials", 30))
    out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = build_feature_sets(config, experiments)
    results = {}

    for name, pack in feature_sets.items():
        columns = pack["columns"]
        x_train, y_train = split_xy(pack["train"], columns)
        x_val, y_val = split_xy(pack["val"], columns)
        x_test, y_test = split_xy(pack["test"], columns)

        print(f"\nFeature set: {name}; dim={x_train.shape[1]}; train={len(y_train)}; val={len(y_val)}; test={len(y_test)}")
        results[name] = {}

        best_xgb = tune_xgboost(x_train, y_train, x_val, y_val, config, n_trials)
        xgb = XGBClassifier(
            **best_xgb,
            objective=xgb_objective(config),
            random_state=42,
            n_jobs=4,
            verbosity=0,
            eval_metric="logloss",
        )
        xgb.fit(x_train, y_train)
        results[name]["xgboost"] = {
            "params": best_xgb,
            "val": evaluate(xgb, x_val, y_val),
            "test": evaluate(xgb, x_test, y_test),
        }

        best_tree = tune_decision_tree(x_train, y_train, x_val, y_val, config, n_trials)
        tree = DecisionTreeClassifier(**best_tree, random_state=42)
        tree.fit(x_train, y_train)
        results[name]["decision_tree"] = {
            "params": best_tree,
            "val": evaluate(tree, x_val, y_val),
            "test": evaluate(tree, x_test, y_test),
            "tree_text": export_text(tree, feature_names=columns, max_depth=4)[:4000],
        }

    out_path = out_dir / "ml_metrics.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"Saved ML metrics -> {out_path}")

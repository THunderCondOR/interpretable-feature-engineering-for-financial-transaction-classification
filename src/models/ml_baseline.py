"""
src/models/ml_baseline.py

Trains XGBoost and DecisionTree on:
- CoT cluster features,
- handcrafted transaction aggregates,
- concatenation of both.

Fixes compared to the initial refactor:
- CoT and handcrafted features are aligned by customer_id, never by row order.
- Rosbank metrics include ROC-AUC, balanced accuracy and macro-F1.
- Split-specific CoT feature files are supported:
    cot_features_train.parquet, cot_features_val.parquet, cot_features_test.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from src.data.loader import load_dataset, add_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _safe_name(s: str) -> str:
    return str(s).replace(" ", "_").replace("/", "_").replace(",", "")[:80]


def _share(part: float, whole: float) -> float:
    return float(part) / float(whole) if whole else 0.0


def build_handcrafted_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    records = []
    for cid, c in df.groupby("customer_id", sort=False):
        label = int(c["label"].iloc[0])
        n_txn = len(c)
        n_days = max(c["tr_datetime"].dt.date.nunique(), 1) if c["tr_datetime"].notna().any() else 1
        pos = c[c["amount"] > 0]["amount"]
        neg = c[c["amount"] < 0]["amount"]
        base = {
            "customer_id": int(cid), "label": label,
            "n_txn": n_txn, "active_days": n_days, "txn_per_day": n_txn / n_days,
            "total_income": float(pos.sum()) if len(pos) else 0.0,
            "total_expense": float(neg.sum()) if len(neg) else 0.0,
            "avg_income": float(pos.mean()) if len(pos) else 0.0,
            "avg_expense": float(neg.mean()) if len(neg) else 0.0,
            "share_income": len(pos) / n_txn if n_txn else 0.0,
            "share_expense": len(neg) / n_txn if n_txn else 0.0,
        }
        for period in ["утро", "день", "вечер", "ночь"]:
            if "period_of_day" in c.columns:
                base[f"share_{period}"] = (c["period_of_day"] == period).mean()
        if "is_weekend" in c.columns:
            base["share_weekend"] = c["is_weekend"].mean()
        for cat, cnt in c.groupby("mcc_code_desc")["amount"].count().to_dict().items():
            safe = _safe_name(cat)
            base[f"cnt_{safe}"] = cnt
        records.append(base)
    return pd.DataFrame(records).fillna(0)


def build_handcrafted_features_rosbank(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for cid, c in df.groupby("customer_id", sort=False):
        label = int(c["label"].iloc[0])
        n_txn = len(c)
        n_days = max(c["tr_datetime"].dt.date.nunique(), 1) if c["tr_datetime"].notna().any() else 1
        n_months = max(c["tr_datetime"].dt.to_period("M").nunique(), 1) if c["tr_datetime"].notna().any() else 1
        base = {
            "customer_id": int(cid), "label": label,
            "n_txn": n_txn, "active_days": n_days, "active_months": n_months,
            "txn_per_day": n_txn / n_days, "txn_per_month": n_txn / n_months,
            "total_amount": float(c["amount"].sum()),
            "avg_amount": float(c["amount"].mean()) if n_txn else 0.0,
            "median_amount": float(c["amount"].median()) if n_txn else 0.0,
            "max_amount": float(c["amount"].max()) if n_txn else 0.0,
            "std_amount": float(c["amount"].std()) if n_txn > 1 else 0.0,
            "n_unique_mcc": c["mcc_code_desc"].nunique(),
            "mcc_diversity": c["mcc_code_desc"].nunique() / max(n_txn, 1),
        }
        if "trx_cat_ru" in c.columns:
            trx = c["trx_cat_ru"].value_counts()
            atm_total = sum(v for k, v in trx.items() if "снятие" in str(k).lower())
            base.update({
                "share_pos": _share(trx.get("оплата картой", 0), n_txn),
                "share_atm": _share(atm_total, n_txn),
                "share_deposit": _share(trx.get("пополнение счета", 0), n_txn),
                "share_c2c_out": _share(trx.get("перевод на карту", 0), n_txn),
                "share_c2c_in": _share(trx.get("входящий перевод с карты", 0), n_txn),
            })
        if "currency_name" in c.columns:
            base["n_currencies"] = c["currency_name"].nunique()
            base["share_rub"] = (c["currency_name"] == "Рубль").mean()
            base["has_foreign_curr"] = int(c["currency_name"].nunique() > 1)
        if "is_weekend" in c.columns:
            base["share_weekend"] = c["is_weekend"].mean()
        if "period_of_day" in c.columns:
            for period in ["утро", "день", "вечер", "ночь"]:
                base[f"share_{period}"] = (c["period_of_day"] == period).mean()
        if c["tr_datetime"].notna().any() and n_txn >= 4:
            c_sorted = c.sort_values("tr_datetime")
            t_min, t_max = c_sorted["tr_datetime"].min(), c_sorted["tr_datetime"].max()
            duration = (t_max - t_min).total_seconds()
            if duration > 0:
                mid = t_min + pd.Timedelta(seconds=duration / 2)
                q1 = t_min + pd.Timedelta(seconds=duration * 0.25)
                q3 = t_max - pd.Timedelta(seconds=duration * 0.25)
                first = (c_sorted["tr_datetime"] <= mid).sum()
                second = (c_sorted["tr_datetime"] > mid).sum()
                early = (c_sorted["tr_datetime"] <= q1).sum()
                recent = (c_sorted["tr_datetime"] >= q3).sum()
                base["second_to_first_txn_ratio"] = second / max(first, 1)
                base["recent_to_early_txn_ratio"] = recent / max(early, 1)
        cat_count = c.groupby("mcc_code_desc")["amount"].count().to_dict()
        cat_amount = c.groupby("mcc_code_desc")["amount"].sum().to_dict()
        for cat, cnt in cat_count.items():
            safe = _safe_name(cat)
            base[f"cnt_{safe}"] = cnt
            base[f"sum_{safe}"] = cat_amount.get(cat, 0)
        records.append(base)
    return pd.DataFrame(records).fillna(0)


def _get_handcrafted_builder(config: dict):
    if config["dataset"]["name"] == "rosbank":
        return lambda df, _cfg: build_handcrafted_features_rosbank(df)
    return build_handcrafted_features


def _eval(model, X, y) -> dict:
    y = np.asarray(y, dtype=int)
    preds = model.predict(X)
    res = {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
        "f1_macro": round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y, preds, average="weighted", zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y, preds)), 4),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
        try:
            res["roc_auc"] = round(float(roc_auc_score(y, model.predict_proba(X)[:, 1])), 4)
        except Exception as exc:
            res["roc_auc_error"] = str(exc)
    return res


def _primary_metric(config: dict) -> str:
    metric = config["dataset"].get("metric", "accuracy")
    if metric in {"roc_auc", "roc_auc_ovr_macro"}:
        return "roc_auc"
    return metric


def _score_for_cv(config: dict) -> str:
    metric = _primary_metric(config)
    return "roc_auc" if metric == "roc_auc" and config["dataset"]["num_labels"] == 2 else "accuracy"


def _tune_xgboost(X_train, y_train, config: dict, n_trials: int = 30):
    n_classes = config["dataset"]["num_labels"]
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    scoring = _score_for_cv(config)

    def objective_fn(trial):
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
        model = XGBClassifier(**params, objective=objective, random_state=42, n_jobs=4, verbosity=0, eval_metric="logloss")
        return cross_val_score(model, X_train, y_train, cv=3, scoring=scoring, n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_decision_tree(X_train, y_train, n_trials: int = 30):
    def objective_fn(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }
        model = DecisionTreeClassifier(**params, random_state=42)
        return cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1).mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _load_cot_split(out_dir: Path, split: str, fallback: Path | None = None) -> pd.DataFrame:
    path = out_dir / f"cot_features_{split}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if fallback and fallback.exists():
        all_df = pd.read_parquet(fallback)
        return all_df
    raise FileNotFoundError(f"CoT features not found for split={split}: {path}")


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"customer_id", "label"}]


def _align_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Force a split-specific feature dataframe to use exactly the train feature schema.

    Missing columns are filled with zeros.
    Extra columns are dropped.
    """
    base_cols = ["customer_id", "label"]
    for col in base_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df.reindex(columns=base_cols + feature_cols, fill_value=0)


def _split_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = df[df["label"] >= 0].copy()
    X = df[feature_cols].values
    y = df["label"].astype(int).values
    return X, y


def run_ml_baseline(config: dict) -> None:
    out_dir = Path(config["output"]["base_dir"])
    n_trials = config.get("optuna", {}).get("n_trials", 30)

    print("Loading splits...")
    train_df = add_features(load_dataset(config, "train"))
    val_df = add_features(load_dataset(config, "val"))
    test_df = add_features(load_dataset(config, "test"))

    print("Building handcrafted features...")
    hc_builder = _get_handcrafted_builder(config)
    hc_train_raw = hc_builder(train_df, config)
    hc_val_raw = hc_builder(val_df, config)
    hc_test_raw = hc_builder(test_df, config)

    print("Loading CoT features...")
    fallback = out_dir / config["output"].get("features", "cot_features.parquet")
    cot_train_raw = _load_cot_split(out_dir, "train", fallback)
    cot_val_raw = _load_cot_split(out_dir, "val", fallback)

    cot_test_path = out_dir / "cot_features_test.parquet"
    cot_test_raw = _load_cot_split(out_dir, "test", fallback) if cot_test_path.exists() else None

    # ------------------------------------------------------------------
    # Fix feature schemas by train split.
    # This is the key fix: val/test must have exactly the same columns
    # as train, otherwise XGBoost crashes with feature shape mismatch.
    # ------------------------------------------------------------------

    hc_cols = _feature_cols(hc_train_raw)
    cot_cols = [c for c in cot_train_raw.columns if c.startswith("cot_cluster_")]

    hc_train = _align_feature_frame(hc_train_raw, hc_cols)
    hc_val = _align_feature_frame(hc_val_raw, hc_cols)
    hc_test = _align_feature_frame(hc_test_raw, hc_cols)

    cot_train = _align_feature_frame(cot_train_raw, cot_cols)
    cot_val = _align_feature_frame(cot_val_raw, cot_cols)
    cot_test = _align_feature_frame(cot_test_raw, cot_cols) if cot_test_raw is not None else None

    def make_concat(cot_df: pd.DataFrame, hc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge CoT and handcrafted features by customer_id and label.
        Never concatenate by row order.
        """
        merged = cot_df.merge(
            hc_df,
            on=["customer_id", "label"],
            how="inner",
            suffixes=("", "_hc"),
        )
        return merged

    cat_train = make_concat(cot_train, hc_train)
    cat_val = make_concat(cot_val, hc_val)
    cat_test = make_concat(cot_test, hc_test) if cot_test is not None else None

    cat_cols = cot_cols + hc_cols

    feature_sets = {
        "cot": {
            "train": cot_train,
            "val": cot_val,
            "test": cot_test,
            "cols": cot_cols,
        },
        "handcrafted": {
            "train": hc_train,
            "val": hc_val,
            "test": hc_test,
            "cols": hc_cols,
        },
        "concat": {
            "train": cat_train,
            "val": cat_val,
            "test": cat_test,
            "cols": cat_cols,
        },
    }

    all_results = {}
    n_classes = config["dataset"]["num_labels"]
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"

    for feat_name, pack in feature_sets.items():
        tr_df = pack["train"]
        va_df = pack["val"]
        te_df = pack["test"]
        feat_cols = pack["cols"]

        X_tr, y_tr = _split_xy(tr_df, feat_cols)
        X_va, y_va = _split_xy(va_df, feat_cols)

        all_results[feat_name] = {}

        print(
            f"\nFeature set: {feat_name}, "
            f"dim={X_tr.shape[1]}, train={len(y_tr)}, val={len(y_va)}"
        )

        if X_tr.shape[1] != X_va.shape[1]:
            raise ValueError(
                f"Internal feature mismatch for {feat_name}: "
                f"train={X_tr.shape[1]}, val={X_va.shape[1]}"
            )

        # XGBoost
        best_xgb = _tune_xgboost(X_tr, y_tr, config, n_trials)
        xgb = XGBClassifier(
            **best_xgb,
            objective=objective,
            random_state=42,
            n_jobs=4,
            verbosity=0,
            eval_metric="logloss",
        )
        xgb.fit(X_tr, y_tr)

        res = {
            "params": best_xgb,
            "val": _eval(xgb, X_va, y_va),
        }

        if te_df is not None and (te_df["label"] >= 0).all():
            X_te, y_te = _split_xy(te_df, feat_cols)

            if X_te.shape[1] != X_tr.shape[1]:
                raise ValueError(
                    f"Internal feature mismatch for {feat_name}: "
                    f"train={X_tr.shape[1]}, test={X_te.shape[1]}"
                )

            res["test"] = _eval(xgb, X_te, y_te)

        all_results[feat_name]["xgboost"] = res

        # Decision Tree
        best_dt = _tune_decision_tree(X_tr, y_tr, n_trials)
        dt = DecisionTreeClassifier(**best_dt, random_state=42)
        dt.fit(X_tr, y_tr)

        dt_res = {
            "params": best_dt,
            "val": _eval(dt, X_va, y_va),
            "tree_text": export_text(
                dt,
                feature_names=feat_cols,
                max_depth=4,
            )[:4000],
        }

        if te_df is not None and (te_df["label"] >= 0).all():
            X_te, y_te = _split_xy(te_df, feat_cols)
            dt_res["test"] = _eval(dt, X_te, y_te)

        all_results[feat_name]["decision_tree"] = dt_res

    out_path = out_dir / "ml_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved -> {out_path}")
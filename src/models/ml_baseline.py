"""
src/models/ml_baseline.py

Обучает XGBoost и Decision Tree на трёх наборах фич:
    cot         — кластерные векторы из атомарных фактов (основной вклад статьи)
    handcrafted — числовые агрегаты из транзакций (экспертные фичи)
    concat      — конкатенация обоих наборов

Результаты сохраняются в results/{dataset}/ml_metrics.json.

Запуск через run_pipeline.py:
    python run_pipeline.py --config configs/gender.yaml --steps ml

Или напрямую:
    from src.models.ml_baseline import run_ml_baseline
    run_ml_baseline(config)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data.loader import load_dataset, add_features


# ---------------------------------------------------------------------------
# Handcrafted features
# ---------------------------------------------------------------------------

def build_handcrafted_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Строит числовой вектор признаков на клиента из транзакций.

    Признаки:
        - базовые агрегаты (n_txn, active_days, txn_per_day, ...)
        - pivot по mcc_code_desc: сумма и количество транзакций на категорию
        - временные паттерны (доля утро/день/вечер/ночь, доля выходных)

    Возвращает DataFrame: customer_id + фичи + label
    """
    records = []

    for cid in df["customer_id"].unique():
        c = df[df["customer_id"] == cid]
        label = int(c["label"].iloc[0])

        # Базовые агрегаты
        n_txn      = len(c)
        n_days     = max(c["tr_datetime"].dt.date.nunique(), 1) \
            if c["tr_datetime"].notna().any() else 1
        pos = c[c["amount"] > 0]["amount"]
        neg = c[c["amount"] < 0]["amount"]

        base = {
            "n_txn":           n_txn,
            "active_days":     n_days,
            "txn_per_day":     round(n_txn / n_days, 3),
            "total_income":    float(pos.sum()) if len(pos) else 0,
            "total_expense":   float(neg.sum()) if len(neg) else 0,
            "avg_income":      float(pos.mean()) if len(pos) else 0,
            "avg_expense":     float(neg.mean()) if len(neg) else 0,
            "share_income":    len(pos) / n_txn if n_txn else 0,
            "share_expense":   len(neg) / n_txn if n_txn else 0,
        }

        # Временные паттерны
        if "period_of_day" in c.columns:
            for period in ["утро", "день", "вечер", "ночь"]:
                base[f"share_{period}"] = (c["period_of_day"] == period).mean()
        if "is_weekend" in c.columns:
            base["share_weekend"] = c["is_weekend"].mean()

        # Pivot: топ-50 категорий по датасету — сумма и количество
        cat_count  = c.groupby("mcc_code_desc")["amount"].count().to_dict()
        cat_amount = c.groupby("mcc_code_desc")["amount"].sum().to_dict()
        for cat, cnt in cat_count.items():
            safe = cat.replace(" ", "_").replace("/", "_").replace(",", "")[:40]
            base[f"cnt_{safe}"]  = cnt
            base[f"sum_{safe}"]  = cat_amount.get(cat, 0)

        base["customer_id"] = int(cid)
        base["label"]       = label
        records.append(base)

    feat_df = pd.DataFrame(records).fillna(0)
    return feat_df


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------

def _tune_xgboost(X_train, y_train, n_trials: int = 30, n_classes: int = 2):
    objective_fn = "binary:logistic" if n_classes == 2 else "multi:softprob"

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
        model = XGBClassifier(
            **params,
            objective=objective_fn,
            num_class=n_classes if n_classes > 2 else None,
            random_state=42, n_jobs=4, verbosity=0,
            use_label_encoder=False, eval_metric="logloss",
        )
        scores = cross_val_score(model, X_train, y_train, cv=3,
                                 scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_decision_tree(X_train, y_train, n_trials: int = 30):
    def objective(trial):
        params = {
            "max_depth":       trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 20),
            "criterion":       trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
        model = DecisionTreeClassifier(**params, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=3,
                                 scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _eval(model, X, y) -> dict:
    preds = model.predict(X)
    return {
        "accuracy":   round(accuracy_score(y, preds), 4),
        "f1_macro":   round(f1_score(y, preds, average="macro",    zero_division=0), 4),
        "f1_weighted":round(f1_score(y, preds, average="weighted", zero_division=0), 4),
    }


def run_ml_baseline(config: dict) -> None:
    """
    Запускает полный ML эксперимент:
        - загружает CoT фичи из cot_features.parquet
        - строит handcrafted фичи из CSV
        - обучает XGBoost и DecisionTree на cot / handcrafted / concat
        - сохраняет метрики и лучшие параметры
    """
    out_dir  = Path(config["output"]["base_dir"])
    feat_path = out_dir / config["output"].get("features", "cot_features.parquet")
    n_trials  = config.get("optuna", {}).get("n_trials", 30)

    # ── Загружаем данные ─────────────────────────────────────────────────────
    print("Загружаем данные...")
    train_df = add_features(load_dataset(config, "train"))
    val_df   = add_features(load_dataset(config, "val"))
    test_df  = add_features(load_dataset(config, "test"))

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # ── CoT фичи ─────────────────────────────────────────────────────────────
    print("Загружаем CoT фичи...")
    cot_df = pd.read_parquet(feat_path)

    def get_split_ids(df):
        return set(df["customer_id"].unique())

    train_ids = get_split_ids(train_df)
    val_ids   = get_split_ids(val_df)
    test_ids  = get_split_ids(test_df)

    def split_cot(cot_df, ids):
        sub = cot_df[cot_df["customer_id"].isin(ids)]
        feat_cols = [c for c in sub.columns if c not in ("customer_id", "label", "features")]
        if "features" in sub.columns:
            X = np.array(sub["features"].tolist())
        else:
            X = sub[feat_cols].values
        y = sub["label"].values
        return X, y

    X_cot_tr, y_tr = split_cot(cot_df, train_ids)
    X_cot_va, y_va = split_cot(cot_df, val_ids)
    X_cot_te, y_te = split_cot(cot_df, test_ids)

    # ── Handcrafted фичи ─────────────────────────────────────────────────────
    print("Строим handcrafted фичи...")
    hc_df = build_handcrafted_features(all_df, config)
    feat_cols_hc = [c for c in hc_df.columns if c not in ("customer_id", "label")]

    def split_hc(ids):
        sub = hc_df[hc_df["customer_id"].isin(ids)]
        return sub[feat_cols_hc].values, sub["label"].values

    X_hc_tr, _    = split_hc(train_ids)
    X_hc_va, _    = split_hc(val_ids)
    X_hc_te, _    = split_hc(test_ids)

    # ── Concat ────────────────────────────────────────────────────────────────
    X_cat_tr = np.hstack([X_cot_tr, X_hc_tr])
    X_cat_va = np.hstack([X_cot_va, X_hc_va])
    X_cat_te = np.hstack([X_cot_te, X_hc_te])

    feature_sets = {
        "cot":         (X_cot_tr, X_cot_va, X_cot_te),
        "handcrafted": (X_hc_tr,  X_hc_va,  X_hc_te),
        "concat":      (X_cat_tr, X_cat_va, X_cat_te),
    }

    n_classes = config["dataset"]["num_labels"]
    all_results = {}

    for feat_name, (X_tr, X_va, X_te) in feature_sets.items():
        print(f"\nФичи: {feat_name} (dim={X_tr.shape[1]})")
        all_results[feat_name] = {}

        # ── XGBoost ───────────────────────────────────────────────────────────
        print(f"  XGBoost — подбор гиперпараметров ({n_trials} trials)...")
        best_xgb = _tune_xgboost(X_tr, y_tr, n_trials, n_classes)
        objective_fn = "binary:logistic" if n_classes == 2 else "multi:softprob"
        xgb = XGBClassifier(
            **best_xgb,
            objective=objective_fn,
            num_class=n_classes if n_classes > 2 else None,
            random_state=42, n_jobs=4, verbosity=0,
            use_label_encoder=False, eval_metric="logloss",
        )
        xgb.fit(X_tr, y_tr)
        xgb_res = {
            "params": best_xgb,
            "val":    _eval(xgb, X_va, y_va),
            "test":   _eval(xgb, X_te, y_te),
        }
        all_results[feat_name]["xgboost"] = xgb_res
        print(f"    val accuracy={xgb_res['val']['accuracy']}  "
              f"test accuracy={xgb_res['test']['accuracy']}")

        # ── Decision Tree ─────────────────────────────────────────────────────
        print(f"  Decision Tree — подбор гиперпараметров ({n_trials} trials)...")
        best_dt = _tune_decision_tree(X_tr, y_tr, n_trials)
        dt = DecisionTreeClassifier(**best_dt, random_state=42)
        dt.fit(X_tr, y_tr)

        # Текстовое представление дерева (для интерпретируемости)
        dt_text = export_text(dt, max_depth=4)

        dt_res = {
            "params":   best_dt,
            "val":      _eval(dt, X_va, y_va),
            "test":     _eval(dt, X_te, y_te),
            "tree_text":dt_text[:2000],  # первые 2000 символов
        }
        all_results[feat_name]["decision_tree"] = dt_res
        print(f"    val accuracy={dt_res['val']['accuracy']}  "
              f"test accuracy={dt_res['test']['accuracy']}")

    # ── Сохранение ────────────────────────────────────────────────────────────
    ml_path = out_dir / "ml_metrics.json"
    with open(ml_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nМетрики сохранены → {ml_path}")

    # Сводная таблица
    print(f"\n{'─'*60}")
    print(f"{'Фичи':<14} {'Модель':<16} {'Val acc':<10} {'Test acc'}")
    print(f"{'─'*60}")
    for feat, models in all_results.items():
        for model_name, res in models.items():
            print(f"{feat:<14} {model_name:<16} "
                  f"{res['val']['accuracy']:<10} {res['test']['accuracy']}")
    print(f"{'─'*60}")
"""
src/models/ml_baseline.py

Обучает XGBoost и Decision Tree на трёх наборах фич:
    cot         — кластерные векторы из атомарных фактов
    handcrafted — числовые агрегаты из транзакций
    concat      — конкатенация обоих наборов

Результаты сохраняются в results/{dataset}/ml_metrics.json.

Запуск через run_pipeline.py:
    python run_pipeline.py --config configs/gender.yaml --steps ml
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
# Handcrafted features — generic (gender / age)
# ---------------------------------------------------------------------------

def build_handcrafted_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Строит числовой вектор признаков на клиента из транзакций.
    Для датасетов где amount может быть как положительным (доход), так и отрицательным (расход).
    """
    records = []

    for cid in df["customer_id"].unique():
        c     = df[df["customer_id"] == cid]
        label = int(c["label"].iloc[0])

        n_txn  = len(c)
        n_days = max(c["tr_datetime"].dt.date.nunique(), 1) \
            if c["tr_datetime"].notna().any() else 1
        pos = c[c["amount"] > 0]["amount"]
        neg = c[c["amount"] < 0]["amount"]

        base = {
            "n_txn":         n_txn,
            "active_days":   n_days,
            "txn_per_day":   round(n_txn / n_days, 3),
            "total_income":  float(pos.sum()) if len(pos) else 0,
            "total_expense": float(neg.sum()) if len(neg) else 0,
            "avg_income":    float(pos.mean()) if len(pos) else 0,
            "avg_expense":   float(neg.mean()) if len(neg) else 0,
            "share_income":  len(pos) / n_txn if n_txn else 0,
            "share_expense": len(neg) / n_txn if n_txn else 0,
        }

        if "period_of_day" in c.columns:
            for period in ["утро", "день", "вечер", "ночь"]:
                base[f"share_{period}"] = (c["period_of_day"] == period).mean()
        if "is_weekend" in c.columns:
            base["share_weekend"] = c["is_weekend"].mean()

        cat_count  = c.groupby("mcc_code_desc")["amount"].count().to_dict()
        cat_amount = c.groupby("mcc_code_desc")["amount"].sum().to_dict()
        for cat, cnt in cat_count.items():
            safe = cat.replace(" ", "_").replace("/", "_").replace(",", "")[:40]
            base[f"cnt_{safe}"] = cnt
            base[f"sum_{safe}"] = cat_amount.get(cat, 0)

        base["customer_id"] = int(cid)
        base["label"]       = label
        records.append(base)

    return pd.DataFrame(records).fillna(0)


# ---------------------------------------------------------------------------
# Handcrafted features — Rosbank-specific
# ---------------------------------------------------------------------------

def build_handcrafted_features_rosbank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит числовой вектор churn-специфичных признаков на клиента.
    amount в rosbank всегда > 0, поэтому income/expense split бессмысленен.
    """
    records = []

    for cid in df["customer_id"].unique():
        c     = df[df["customer_id"] == cid]
        label = int(c["label"].iloc[0])
        n_txn = len(c)

        n_days   = max(c["tr_datetime"].dt.date.nunique(), 1) \
            if c["tr_datetime"].notna().any() else 1
        n_months = max(c["tr_datetime"].dt.to_period("M").nunique(), 1) \
            if c["tr_datetime"].notna().any() else 1

        base = {
            "n_txn":          n_txn,
            "active_days":    n_days,
            "active_months":  n_months,
            "txn_per_day":    round(n_txn / n_days, 3),
            "txn_per_month":  round(n_txn / n_months, 3),
            "total_amount":   float(c["amount"].sum()),
            "avg_amount":     float(c["amount"].mean()),
            "median_amount":  float(c["amount"].median()),
            "max_amount":     float(c["amount"].max()),
            "std_amount":     float(c["amount"].std()) if n_txn > 1 else 0,
        }

        if "trx_cat_ru" in c.columns:
            trx_counts = c["trx_cat_ru"].value_counts()
            atm_total  = sum(v for k, v in trx_counts.items() if "снятие" in k)
            base["share_pos"]     = trx_counts.get("оплата картой", 0) / n_txn
            base["share_atm"]     = atm_total / n_txn
            base["share_deposit"] = trx_counts.get("пополнение счёта", 0) / n_txn
            base["share_c2c_out"] = trx_counts.get("перевод на карту", 0) / n_txn
            base["share_c2c_in"]  = trx_counts.get("входящий перевод с карты", 0) / n_txn
            base["has_c2c_out"]   = int(base["share_c2c_out"] > 0)
            base["has_deposit"]   = int(base["share_deposit"] > 0)

        if "currency_name" in c.columns:
            base["n_currencies"]     = c["currency_name"].nunique()
            base["share_rub"]        = (c["currency_name"] == "Рубль").mean()
            base["has_foreign_curr"] = int(c["currency_name"].nunique() > 1)

        base["n_unique_mcc"]  = c["mcc_code_desc"].nunique()
        base["mcc_diversity"] = round(c["mcc_code_desc"].nunique() / n_txn, 4)

        if c["tr_datetime"].notna().any():
            base["share_weekend"] = c["is_weekend"].mean() \
                if "is_weekend" in c.columns else 0
            for period in ["утро", "день", "вечер", "ночь"]:
                base[f"share_{period}"] = (c["period_of_day"] == period).mean() \
                    if "period_of_day" in c.columns else 0

            # Recency: среднее число транзакций в последней четверти периода
            # vs первой четверти (по времени, не по count)
            if n_txn >= 4 and c["tr_datetime"].notna().any():
                c_sorted  = c.sort_values("tr_datetime")
                t_min = c_sorted["tr_datetime"].min()
                t_max = c_sorted["tr_datetime"].max()
                duration  = (t_max - t_min).total_seconds()
                if duration > 0:
                    cutoff_recent = t_max  - pd.Timedelta(seconds=duration * 0.25)
                    cutoff_old    = t_min  + pd.Timedelta(seconds=duration * 0.25)
                    n_recent = (c_sorted["tr_datetime"] >= cutoff_recent).sum()
                    n_old    = (c_sorted["tr_datetime"] <= cutoff_old).sum()
                    base["recency_ratio"] = round(n_recent / max(n_old, 1), 3)
                else:
                    base["recency_ratio"] = 1.0
            else:
                base["recency_ratio"] = 1.0

        cat_count  = c.groupby("mcc_code_desc")["amount"].count().to_dict()
        cat_amount = c.groupby("mcc_code_desc")["amount"].sum().to_dict()
        for cat, cnt in cat_count.items():
            safe = cat.replace(" ", "_").replace("/", "_").replace(",", "")[:40]
            base[f"cnt_{safe}"] = cnt
            base[f"sum_{safe}"] = cat_amount.get(cat, 0)

        base["customer_id"] = int(cid)
        base["label"]       = label
        records.append(base)

    return pd.DataFrame(records).fillna(0)


def _get_handcrafted_builder(config: dict):
    """Диспетчер: вернуть нужную функцию построения handcrafted фич."""
    if config["dataset"]["name"] == "rosbank":
        return lambda df, _cfg: build_handcrafted_features_rosbank(df)
    return build_handcrafted_features


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
            eval_metric="logloss",
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
            "max_depth":        trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "criterion":        trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
        model = DecisionTreeClassifier(**params, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=3,
                                 scoring="accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _eval(model, X, y) -> dict:
    preds = model.predict(X)
    return {
        "accuracy":    round(accuracy_score(y, preds), 4),
        "f1_macro":    round(f1_score(y, preds, average="macro",    zero_division=0), 4),
        "f1_weighted": round(f1_score(y, preds, average="weighted", zero_division=0), 4),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_ml_baseline(config: dict) -> None:
    """
    Запускает полный ML эксперимент:
        - загружает CoT фичи из cot_features.parquet
        - строит handcrafted фичи из CSV (с диспетчингом для rosbank)
        - обучает XGBoost и DecisionTree на cot / handcrafted / concat
        - сохраняет метрики
    """
    out_dir   = Path(config["output"]["base_dir"])
    feat_path = out_dir / config["output"].get("features", "cot_features.parquet")
    n_trials  = config.get("optuna", {}).get("n_trials", 30)

    print("Загружаем данные...")
    train_df = add_features(load_dataset(config, "train"))
    val_df   = add_features(load_dataset(config, "val"))
    test_df  = add_features(load_dataset(config, "test"))
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("Загружаем CoT фичи...")
    cot_df = pd.read_parquet(feat_path)

    train_ids = set(train_df["customer_id"].unique())
    val_ids   = set(val_df["customer_id"].unique())
    test_ids  = set(test_df["customer_id"].unique())

    def split_cot(ids):
        sub = cot_df[cot_df["customer_id"].isin(ids)]
        if "features" in sub.columns:
            X = np.array(sub["features"].tolist())
        else:
            feat_cols = [c for c in sub.columns if c not in ("customer_id", "label")]
            X = sub[feat_cols].values
        y = sub["label"].values
        return X, y

    X_cot_tr, y_tr = split_cot(train_ids)
    X_cot_va, y_va = split_cot(val_ids)
    X_cot_te, y_te = split_cot(test_ids)

    print("Строим handcrafted фичи...")
    hc_builder  = _get_handcrafted_builder(config)
    hc_df       = hc_builder(all_df, config)
    feat_cols_hc = [c for c in hc_df.columns if c not in ("customer_id", "label")]

    def split_hc(ids):
        sub = hc_df[hc_df["customer_id"].isin(ids)]
        return sub[feat_cols_hc].values, sub["label"].values

    X_hc_tr, _ = split_hc(train_ids)
    X_hc_va, _ = split_hc(val_ids)
    X_hc_te, _ = split_hc(test_ids)

    X_cat_tr = np.hstack([X_cot_tr, X_hc_tr])
    X_cat_va = np.hstack([X_cot_va, X_hc_va])
    X_cat_te = np.hstack([X_cot_te, X_hc_te])

    feature_sets = {
        "cot":         (X_cot_tr, X_cot_va, X_cot_te),
        "handcrafted": (X_hc_tr,  X_hc_va,  X_hc_te),
        "concat":      (X_cat_tr, X_cat_va, X_cat_te),
    }

    n_classes   = config["dataset"]["num_labels"]
    all_results = {}

    for feat_name, (X_tr, X_va, X_te) in feature_sets.items():
        print(f"\nФичи: {feat_name} (dim={X_tr.shape[1]})")
        all_results[feat_name] = {}

        # XGBoost
        print(f"  XGBoost — подбор гиперпараметров ({n_trials} trials)...")
        best_xgb     = _tune_xgboost(X_tr, y_tr, n_trials, n_classes)
        objective_fn = "binary:logistic" if n_classes == 2 else "multi:softprob"
        xgb = XGBClassifier(
            **best_xgb,
            objective=objective_fn,
            num_class=n_classes if n_classes > 2 else None,
            random_state=42, n_jobs=4, verbosity=0,
            eval_metric="logloss",
        )
        xgb.fit(X_tr, y_tr)
        xgb_res = {
            "params": best_xgb,
            "val":    _eval(xgb, X_va, y_va),
            "test":   _eval(xgb, X_te, y_te),
        }
        all_results[feat_name]["xgboost"] = xgb_res
        print(f"    val acc={xgb_res['val']['accuracy']}  "
              f"test acc={xgb_res['test']['accuracy']}")

        # Decision Tree
        print(f"  Decision Tree — подбор гиперпараметров ({n_trials} trials)...")
        best_dt = _tune_decision_tree(X_tr, y_tr, n_trials)
        dt = DecisionTreeClassifier(**best_dt, random_state=42)
        dt.fit(X_tr, y_tr)
        dt_text = export_text(dt, max_depth=4)
        dt_res  = {
            "params":    best_dt,
            "val":       _eval(dt, X_va, y_va),
            "test":      _eval(dt, X_te, y_te),
            "tree_text": dt_text[:2000],
        }
        all_results[feat_name]["decision_tree"] = dt_res
        print(f"    val acc={dt_res['val']['accuracy']}  "
              f"test acc={dt_res['test']['accuracy']}")

    ml_path = out_dir / "ml_metrics.json"
    with open(ml_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nМетрики сохранены → {ml_path}")

    print(f"\n{'─'*60}")
    print(f"{'Фичи':<14} {'Модель':<16} {'Val acc':<10} {'Test acc'}")
    print(f"{'─'*60}")
    for feat, models in all_results.items():
        for model_name, res in models.items():
            print(f"{feat:<14} {model_name:<16} "
                  f"{res['val']['accuracy']:<10} {res['test']['accuracy']}")
    print(f"{'─'*60}")
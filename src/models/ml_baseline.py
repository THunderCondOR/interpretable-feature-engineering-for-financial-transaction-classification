"""Classical ML baselines for prepared transaction datasets.

Supported feature sets:
- standard: generic amount aggregates and category pivots;
- handcrafted: deterministic transaction aggregates;
- cot: train-fitted CoT cluster features;
- concat: handcrafted + CoT features merged by customer_id and label.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from src.data.loader import add_features, load_dataset
from src.experiments.artifacts import files_fingerprint, stage_signature

optuna.logging.set_verbosity(optuna.logging.WARNING)

EXPERIMENTS = {"standard", "handcrafted", "cot", "concat"}


def safe_name(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_").replace(",", "")[:100]


def share(part: float, whole: float) -> float:
    return float(part) / float(whole) if whole else 0.0


def build_handcrafted_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if config["dataset"]["name"] == "rosbank":
        return build_rosbank_handcrafted_features(df)
    return build_generic_handcrafted_features(
        df,
        amount_semantics=config.get("dataset", {}).get(
            "amount_semantics", "signed_cashflow"
        ),
    )


def build_standard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build label-agnostic transaction aggregates without behavioral heuristics."""
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
        dates = df.assign(
            _day=df["tr_datetime"].dt.date,
            _month=df["tr_datetime"].dt.to_period("M"),
        )
        active_days = dates.groupby("customer_id")["_day"].nunique().rename("active_days")
        active_months = dates.groupby("customer_id")["_month"].nunique().rename("active_months")
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
        pivot = (
            df.groupby(["customer_id", "mcc_code_desc"])["amount"]
            .agg(agg_func)
            .unstack("mcc_code_desc")
            .fillna(0)
        )
        pivot.columns = [f"mcc_{safe_name(category)}_{suffix}" for category in pivot.columns]
        pivots.append(pivot)

    features = pd.concat([base] + pivots, axis=1).fillna(0)
    return labels.join(features, how="left").fillna(0).reset_index()


def build_generic_handcrafted_features(
    df: pd.DataFrame,
    *,
    amount_semantics: str = "signed_cashflow",
) -> pd.DataFrame:
    records = []
    for customer_id, client in df.groupby("customer_id", sort=False):
        label = int(client["label"].iloc[0])
        n_txn = len(client)
        n_days = max(client["tr_datetime"].dt.date.nunique(), 1) if client["tr_datetime"].notna().any() else 1
        pos = client.loc[client["amount"] > 0, "amount"]
        neg = client.loc[client["amount"] < 0, "amount"]
        row: dict[str, float | int] = {
            "customer_id": int(customer_id),
            "label": label,
            "n_txn": n_txn,
            "active_days": n_days,
            "txn_per_day": n_txn / n_days,
        }
        if amount_semantics == "unsigned_transaction_value":
            amounts = client["amount"]
            row.update(
                {
                    "total_transaction_value": float(amounts.sum()),
                    "avg_transaction_value": float(amounts.mean()) if n_txn else 0.0,
                    "median_transaction_value": float(amounts.median()) if n_txn else 0.0,
                    "max_transaction_value": float(amounts.max()) if n_txn else 0.0,
                    "std_transaction_value": float(amounts.std()) if n_txn > 1 else 0.0,
                }
            )
        else:
            row.update(
                {
                    "total_income": float(pos.sum()) if len(pos) else 0.0,
                    "total_expense": float(neg.sum()) if len(neg) else 0.0,
                    "avg_income": float(pos.mean()) if len(pos) else 0.0,
                    "avg_expense": float(neg.mean()) if len(neg) else 0.0,
                    "share_income": share(len(pos), n_txn),
                    "share_expense": share(len(neg), n_txn),
                }
            )
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
    x = df[columns].replace([np.inf, -np.inf], 0).fillna(0).to_numpy(dtype=np.float32, copy=True)
    y = df["label"].astype(int).to_numpy()
    return x, y


def load_cot_features(out_dir: Path, split: str) -> pd.DataFrame:
    path = out_dir / f"cot_features_{split}.parquet"
    return pd.read_parquet(path)


def canonical_client_frame(transactions: pd.DataFrame, split: str) -> pd.DataFrame:
    label_counts = transactions.groupby("customer_id")["label"].nunique()
    if (label_counts != 1).any():
        raise ValueError(f"{split} contains clients with inconsistent labels")
    return (
        transactions[["customer_id", "label"]]
        .drop_duplicates("customer_id")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )


def validate_client_feature_frame(
    frame: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    split: str,
    source: str,
) -> pd.DataFrame:
    if not {"customer_id", "label"} <= set(frame.columns):
        raise ValueError(f"{source} {split} lacks customer_id/label columns")
    if frame["customer_id"].duplicated().any():
        raise ValueError(f"{source} {split} contains duplicate customer IDs")
    expected_ids = set(canonical["customer_id"])
    actual_ids = set(frame["customer_id"])
    if actual_ids != expected_ids:
        raise ValueError(
            f"{source} {split} client IDs differ from canonical split: "
            f"missing={len(expected_ids - actual_ids)}, extra={len(actual_ids - expected_ids)}"
        )
    expected_labels = canonical.set_index("customer_id")["label"]
    actual_labels = frame.set_index("customer_id")["label"].reindex(expected_labels.index)
    if not actual_labels.equals(expected_labels):
        raise ValueError(f"{source} {split} labels differ from canonical split")
    return canonical[["customer_id"]].merge(
        frame,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )


def merge_feature_frames(cot: pd.DataFrame, handcrafted: pd.DataFrame) -> pd.DataFrame:
    if cot["customer_id"].duplicated().any() or handcrafted["customer_id"].duplicated().any():
        raise ValueError("Cannot concatenate feature frames with duplicate customer IDs")
    merged = cot.merge(
        handcrafted,
        on="customer_id",
        how="outer",
        suffixes=("", "_hc"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("CoT and handcrafted feature frames contain different client IDs")
    if not merged["label"].equals(merged["label_hc"]):
        raise ValueError("CoT and handcrafted feature frames contain different labels")
    return merged.drop(columns=["label_hc", "_merge"])


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


def tune_xgboost(x_train, y_train, x_val, y_val, config: dict, n_trials: int, seed: int = 17) -> dict:
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
            random_state=seed,
            n_jobs=2,
            verbosity=0,
            eval_metric="logloss",
            tree_method="hist",
        )
        model.fit(x_train, y_train)
        return primary_score(evaluate(model, x_val, y_val), config)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_decision_tree(x_train, y_train, x_val, y_val, config: dict, n_trials: int, seed: int = 17) -> dict:
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        }
        model = DecisionTreeClassifier(**params, random_state=seed)
        model.fit(x_train, y_train)
        return primary_score(evaluate(model, x_val, y_val), config)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def build_feature_sets(config: dict, experiments: list[str]) -> dict[str, dict]:
    requested = set(experiments)
    unknown = requested - EXPERIMENTS
    if unknown:
        raise ValueError(f"Unknown ML experiments: {sorted(unknown)}")

    out_dir = Path(config["output"]["base_dir"])
    cot_dir = Path(
        config.get("input", {}).get("cot_features_base_dir", out_dir)
    )
    train_df = add_features(load_dataset(config, "train"))
    val_df = add_features(load_dataset(config, "val"))
    test_df = add_features(load_dataset(config, "test"))
    canonical_clients = {
        "train": canonical_client_frame(train_df, "train"),
        "val": canonical_client_frame(val_df, "val"),
        "test": canonical_client_frame(test_df, "test"),
    }

    feature_sets = {}
    if "standard" in requested:
        standard_train_raw = build_standard_features(train_df)
        standard_val_raw = build_standard_features(val_df)
        standard_test_raw = build_standard_features(test_df)
        standard_cols = feature_columns(standard_train_raw)
        feature_sets["standard"] = {
            "train": align_features(standard_train_raw, standard_cols),
            "val": align_features(standard_val_raw, standard_cols),
            "test": align_features(standard_test_raw, standard_cols),
            "columns": standard_cols,
        }

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
        cot_train_raw = validate_client_feature_frame(
            load_cot_features(cot_dir, "train"),
            canonical_clients["train"],
            split="train",
            source="CoT features",
        )
        cot_val_raw = validate_client_feature_frame(
            load_cot_features(cot_dir, "val"),
            canonical_clients["val"],
            split="val",
            source="CoT features",
        )
        cot_test_raw = validate_client_feature_frame(
            load_cot_features(cot_dir, "test"),
            canonical_clients["test"],
            split="test",
            source="CoT features",
        )
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


def _seed_metric_summary(runs: dict[str, dict], split: str) -> dict:
    metric_names = sorted({
        metric
        for run in runs.values()
        for metric, value in run[split].items()
        if isinstance(value, (int, float)) and metric != "n"
    })
    summary = {}
    for metric in metric_names:
        values = np.asarray([run[split][metric] for run in runs.values()], dtype=float)
        summary[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _balanced_accuracy_ci(y_true, y_pred, *, samples=1000, seed=17):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    values = []
    for _ in range(samples):
        indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True)
            for group in groups
        ])
        values.append(balanced_accuracy_score(y_true[indices], y_pred[indices]))
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "method": "paired_stratified_client_bootstrap",
    }


def prediction_artifact_complete(
    records: list[dict],
    *,
    feature_set: str,
    seeds: list[int],
    frames: dict[str, pd.DataFrame],
    num_labels: int,
) -> bool:
    probability_fields = {f"probability_{label}" for label in range(num_labels)}
    for classifier in ("xgboost", "decision_tree"):
        for seed in seeds:
            for split, frame in frames.items():
                cell = [
                    row for row in records
                    if row.get("feature_set") == feature_set
                    and row.get("classifier") == classifier
                    and row.get("seed") == seed
                    and row.get("split") == split
                ]
                expected_labels = {
                    int(row.customer_id): int(row.label)
                    for row in frame[["customer_id", "label"]].itertuples(index=False)
                }
                if len(cell) != len(expected_labels):
                    return False
                ids = [int(row.get("customer_id", -1)) for row in cell]
                if len(ids) != len(set(ids)) or set(ids) != set(expected_labels):
                    return False
                for row in cell:
                    customer_id = int(row.get("customer_id", -1))
                    if int(row.get("label", -1)) != expected_labels[customer_id]:
                        return False
                    if int(row.get("prediction", -1)) not in range(num_labels):
                        return False
                    row_probability_fields = {
                        key for key in row if key.startswith("probability_")
                    }
                    if row_probability_fields != probability_fields:
                        return False
                    probabilities = np.asarray(
                        [row[field] for field in sorted(probability_fields)],
                        dtype=float,
                    )
                    if (
                        not np.isfinite(probabilities).all()
                        or (probabilities < 0).any()
                        or (probabilities > 1).any()
                        or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)
                    ):
                        return False
    return True


def run_ml_baseline(config: dict, experiments: list[str] | None = None) -> None:
    experiments = experiments or ["handcrafted", "cot", "concat"]
    n_trials = int(config.get("optuna", {}).get("n_trials", 30))
    seeds = [
        int(seed) for seed in config.get("evaluation", {}).get(
            "seeds",
            [config.get("experiment", {}).get("seed", 17)],
        )
    ]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("evaluation.seeds must contain unique integer seeds")
    bootstrap_samples = int(config.get("evaluation", {}).get("bootstrap_samples", 1000))
    out_dir = Path(config["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ml_metrics.json"
    prediction_path = out_dir / "ml_predictions.jsonl"

    results = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as file:
            existing = json.load(file)
        if isinstance(existing, dict):
            results.update(existing)

    prediction_records = []
    if prediction_path.exists():
        with open(prediction_path, encoding="utf-8") as file:
            prediction_records = [json.loads(line) for line in file if line.strip()]

    def save_results() -> None:
        temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
        temp_path.replace(out_path)
        prediction_temp = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
        with open(prediction_temp, "w", encoding="utf-8") as file:
            for record in prediction_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        prediction_temp.replace(prediction_path)
        print(f"Saved seeded ML metrics -> {out_path}")

    for name in experiments:
        signature = ml_artifact_signature(config, name)
        expected_seed_keys = {str(seed) for seed in seeds}
        feature_sets = build_feature_sets(config, [name])
        pack = feature_sets[name]
        metrics_complete = all(
            set(results.get(name, {}).get(classifier, {}).get("runs", {}))
            == expected_seed_keys
            for classifier in ("xgboost", "decision_tree")
        )
        predictions_complete = prediction_artifact_complete(
            prediction_records,
            feature_set=name,
            seeds=seeds,
            frames={split: pack[split] for split in ("train", "val", "test")},
            num_labels=int(config["dataset"]["num_labels"]),
        )
        if (
            name in results
            and results[name].get("artifact_signature") == signature
            and metrics_complete
            and predictions_complete
        ):
            print(f"\nFeature set: {name}; compatible seeded result already complete, skipping")
            del feature_sets, pack
            continue
        columns = pack["columns"]
        x_train, y_train = split_xy(pack["train"], columns)
        x_val, y_val = split_xy(pack["val"], columns)
        x_test, y_test = split_xy(pack["test"], columns)
        print(
            f"\nFeature set: {name}; dim={x_train.shape[1]}; "
            f"train={len(y_train)}; val={len(y_val)}; test={len(y_test)}; seeds={seeds}"
        )

        tuning_seed = seeds[0]
        best_xgb = tune_xgboost(
            x_train, y_train, x_val, y_val, config, n_trials, seed=tuning_seed
        )
        best_tree = tune_decision_tree(
            x_train, y_train, x_val, y_val, config, n_trials, seed=tuning_seed
        )
        classifier_results = {
            "xgboost": {"params": best_xgb, "runs": {}},
            "decision_tree": {"params": best_tree, "runs": {}},
        }
        prediction_records = [
            row for row in prediction_records if row.get("feature_set") != name
        ]

        for seed in seeds:
            classifiers = {
                "xgboost": XGBClassifier(
                    **best_xgb,
                    objective=xgb_objective(config),
                    random_state=seed,
                    n_jobs=2,
                    verbosity=0,
                    eval_metric="logloss",
                    tree_method="hist",
                ),
                "decision_tree": DecisionTreeClassifier(
                    **best_tree,
                    random_state=seed,
                ),
            }
            for classifier_name, model in classifiers.items():
                model.fit(x_train, y_train)
                split_values = {
                    "train": (pack["train"], x_train, y_train),
                    "val": (pack["val"], x_val, y_val),
                    "test": (pack["test"], x_test, y_test),
                }
                split_metrics = {}
                for split_name, (split_frame, values, truth) in split_values.items():
                    metrics = evaluate(model, values, truth)
                    predictions = model.predict(values)
                    if split_name == "test":
                        metrics["balanced_accuracy_ci"] = _balanced_accuracy_ci(
                            truth,
                            predictions,
                            samples=bootstrap_samples,
                            seed=seed,
                        )
                    split_metrics[split_name] = metrics
                    probabilities = (
                        model.predict_proba(values)
                        if hasattr(model, "predict_proba")
                        else None
                    )
                    for index, customer_id in enumerate(split_frame["customer_id"]):
                        record = {
                            "feature_set": name,
                            "classifier": classifier_name,
                            "seed": seed,
                            "split": split_name,
                            "customer_id": int(customer_id),
                            "label": int(truth[index]),
                            "prediction": int(predictions[index]),
                        }
                        if probabilities is not None:
                            record.update({
                                f"probability_{int(class_id)}": float(
                                    probabilities[index, probability_index]
                                )
                                for probability_index, class_id in enumerate(model.classes_)
                            })
                        prediction_records.append(record)
                classifier_results[classifier_name]["runs"][str(seed)] = {
                    "val": split_metrics["val"],
                    "test": split_metrics["test"],
                }

        for classifier_name, classifier_result in classifier_results.items():
            runs = classifier_result["runs"]
            primary = runs[str(seeds[0])]
            classifier_result["val"] = primary["val"]
            classifier_result["test"] = primary["test"]
            classifier_result["summary"] = {
                "val": _seed_metric_summary(runs, "val"),
                "test": _seed_metric_summary(runs, "test"),
            }
        tree_for_text = DecisionTreeClassifier(**best_tree, random_state=seeds[0])
        tree_for_text.fit(x_train, y_train)
        classifier_results["decision_tree"]["tree_text"] = export_text(
            tree_for_text,
            feature_names=columns,
            max_depth=4,
        )[:4000]
        results[name] = {
            "artifact_signature": signature,
            "seeds": seeds,
            **classifier_results,
        }
        save_results()

        del feature_sets, pack, x_train, y_train, x_val, y_val, x_test, y_test
        gc.collect()


def ml_artifact_signature(config: dict, feature_set: str) -> str:
    """Identify every source that can change an ML result.

    Old metrics without this signature are deliberately treated as legacy and are
    recomputed rather than silently reused.
    """
    source_paths = list(config["dataset"]["splits"].values())
    if feature_set in {"cot", "concat"}:
        out_dir = Path(config.get("input", {}).get(
            "cot_features_base_dir", config["output"]["base_dir"]
        ))
        source_paths.extend(
            out_dir / f"cot_features_{split}.parquet"
            for split in ("train", "val", "test")
        )
    relevant_config = {
        "dataset": config.get("dataset", {}),
        "pipeline": config.get("pipeline", {}),
        "optuna": config.get("optuna", {}),
        "evaluation": config.get("evaluation", {}),
        "input": config.get("input", {}),
        "experiment": config.get("experiment", {}),
        "feature_set": feature_set,
    }
    return stage_signature(
        "ml_baseline",
        inputs=files_fingerprint(source_paths),
        configuration=relevant_config,
    )

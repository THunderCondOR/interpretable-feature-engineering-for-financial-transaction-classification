"""Build a publication figure comparing TreeSHAP feature languages.

The same attribution mechanism is applied to a conventional aggregate model
and a semantic claim-cluster model.  XGBoost ``pred_contribs`` implements
TreeSHAP, so the script does not require the optional ``shap`` package.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

from interpretable_features.experiments.artifacts import atomic_write_json
from interpretable_features.models.ml_baseline import build_feature_sets, split_xy


def readable_claim(text: str, width: int = 43) -> str:
    value = " ".join(str(text).split())
    value = value.removeprefix("the client’s ").removeprefix("the client's ")
    value = value.removeprefix("the client ")
    if value:
        value = value[0].upper() + value[1:]
    return textwrap.fill(value, width=width)


def raw_label(name: str, width: int = 32) -> str:
    exact = {
        "calendar_span_days": "Observation-window length",
        "txn_per_day": "Transactions per active day",
        "active_months": "Number of active months",
        "amount_p05": "5th percentile of transaction value",
        "mcc_Снятие_наличных_[пополнение_счета]_sum": "Total value of cash account top-ups",
        "mcc_Снятие_наличных_[снятие_наличных_(банкомат_Росбанка)]_count": "Number of cash withdrawals at Rosbank ATMs",
        "mcc_Снятие_наличных_[снятие_наличных_(банкомат_Росбанка)]_min": "Smallest cash withdrawal at Rosbank ATMs",
        "mcc_Снятие_наличных_[снятие_наличных_(другой_банк)]_sum": "Total cash withdrawals from other banks",
        "mcc_Снятие_наличных_[снятие_наличных_(другой_банк)]_count": "Number of cash withdrawals from other banks",
        "mcc_Фастфуд_[оплата_картой]_sum": "Total card spending at fast-food merchants",
        "mcc_Аптеки_[оплата_картой]_max": "Largest card payment at pharmacies",
        "mcc_Аптеки_[оплата_картой]_sum": "Total card spending at pharmacies",
        "mcc_Продовольственные_магазины_[оплата_картой]_mean": "Average card payment at grocery stores",
        "mcc_Продовольственные_магазины_[оплата_картой]_median": "Median card payment at grocery stores",
        "mcc_Цветы_[оплата_картой]_max": "Largest card payment at flower shops",
        "mcc_Разные_специализированные_магазины_[оплата_картой]_count": "Number of card payments at specialty stores",
        "mcc_Разные_специализированные_магазины_[оплата_картой]_sum": "Total card spending at specialty stores",
        "mcc_Косметика_[оплата_картой]_min": "Smallest card payment at cosmetics stores",
    }
    value = exact.get(str(name), str(name).replace("_", " "))
    return textwrap.fill(value, width=width)


def choose_local_example(
    customer_ids: list[str],
    truth: np.ndarray,
    baseline_probability: np.ndarray,
    claim_probability: np.ndarray,
    baseline_contributions: np.ndarray | None = None,
    claim_contributions: np.ndarray | None = None,
    *,
    minimum_confidence: float = 0.80,
) -> tuple[int, dict[str, Any]]:
    """Choose a pre-specified median-fidelity jointly correct example."""
    baseline_prediction = (baseline_probability >= 0.5).astype(int)
    claim_prediction = (claim_probability >= 0.5).astype(int)
    baseline_confidence = np.maximum(baseline_probability, 1 - baseline_probability)
    claim_confidence = np.maximum(claim_probability, 1 - claim_probability)
    joint_correct = (baseline_prediction == truth) & (claim_prediction == truth)
    eligible = (
        joint_correct
        & (baseline_confidence >= minimum_confidence)
        & (claim_confidence >= minimum_confidence)
    )
    signed_requirement = baseline_contributions is not None and claim_contributions is not None
    if signed_requirement:
        baseline_signed = (baseline_contributions > 1e-8).any(axis=1) & (
            baseline_contributions < -1e-8
        ).any(axis=1)
        claim_signed = (claim_contributions > 1e-8).any(axis=1) & (claim_contributions < -1e-8).any(
            axis=1
        )
        eligible &= baseline_signed & claim_signed
    pool = np.flatnonzero(eligible)
    fallback = False
    if not len(pool):
        pool = np.flatnonzero(joint_correct)
        fallback = True
    if not len(pool):
        pool = np.arange(len(truth))
        fallback = True
    gap = np.abs(baseline_probability - claim_probability)
    median_gap = float(np.median(gap[pool]))
    index = min(
        pool.tolist(),
        key=lambda item: (abs(float(gap[item]) - median_gap), customer_ids[item]),
    )
    return int(index), {
        "rule": (
            "jointly correct; both predicted-class confidences >= 0.80; "
            "both local explanations contain positive and negative contributions; "
            "probability-gap nearest eligible median"
            if signed_requirement
            else "jointly correct; both predicted-class confidences >= 0.80; "
            "probability-gap nearest eligible median"
        ),
        "fallback_used": fallback,
        "eligible_clients": int(len(pool)),
        "median_probability_gap": median_gap,
        "selected_probability_gap": float(gap[index]),
    }


def fit_model(
    config: dict[str, Any],
    params: dict[str, Any],
    values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> XGBClassifier:
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(values, labels)
    return model


def tree_shap(model: XGBClassifier, values: np.ndarray) -> np.ndarray:
    contribution = model.get_booster().predict(xgb.DMatrix(values), pred_contribs=True)
    # The final column is the expected-value bias term.
    return np.asarray(contribution[:, :-1], dtype=float)


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    importance = np.abs(values).mean(axis=0)
    return np.argsort(importance)[-min(count, len(importance)) :]


def contribution_panel(
    axis: plt.Axes,
    values: np.ndarray,
    labels: list[str],
    indices: np.ndarray,
    *,
    title: str,
    color: str,
) -> None:
    scores = np.abs(values).mean(axis=0)[indices]
    axis.barh(np.arange(len(indices)), scores, color=color, alpha=0.92)
    axis.set_yticks(np.arange(len(indices)), [labels[index] for index in indices])
    axis.set_xlabel("mean |TreeSHAP value|")
    axis.set_title(title, loc="left", fontweight="bold", pad=12)
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)


def local_panel(
    axis: plt.Axes,
    row: np.ndarray,
    labels: list[str],
    *,
    title: str,
    target_name: str,
) -> None:
    count = min(7, len(row))
    positive = np.flatnonzero(row > 1e-8)
    negative = np.flatnonzero(row < -1e-8)
    if len(positive) and len(negative):
        n_negative = min(3, len(negative))
        n_positive = min(count - n_negative, len(positive))
        # If one side has too few features, fill the remaining slots from the
        # other while still guaranteeing that both directions are displayed.
        remaining = count - n_negative - n_positive
        if remaining and len(negative) > n_negative:
            n_negative += min(remaining, len(negative) - n_negative)
        negative_indices = negative[np.argsort(np.abs(row[negative]))[-n_negative:]]
        positive_indices = positive[np.argsort(np.abs(row[positive]))[-n_positive:]]
        indices = np.asarray(
            sorted(
                [*negative_indices.tolist(), *positive_indices.tolist()],
                key=lambda index: row[index],
            )
        )
    else:
        indices = np.argsort(np.abs(row))[-count:]
    selected = row[indices]
    colors = np.where(selected >= 0, "#E45756", "#4C78A8")
    axis.barh(np.arange(len(indices)), selected, color=colors, alpha=0.92)
    axis.set_yticks(np.arange(len(indices)), [labels[index] for index in indices])
    axis.axvline(0, color="#31343A", linewidth=0.8)
    limit = max(float(np.abs(selected).max()) * 1.08, 1e-6)
    axis.set_xlim(-limit, limit)
    axis.set_xlabel(f"TreeSHAP contribution to {target_name} log-odds")
    axis.set_title(title, loc="left", fontweight="bold", pad=12)
    axis.grid(axis="x", alpha=0.18)
    axis.spines[["top", "right", "left"]].set_visible(False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        type=Path,
        required=True,
    )
    parser.add_argument("--baseline", default="standard")
    parser.add_argument("--claims", default="cot")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top", type=int, default=9)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = {
        "cell": str(args.cell),
        "baseline": args.baseline,
        "claims": args.claims,
        "seed": args.seed,
        "output": str(args.output),
        "method": "XGBoost TreeSHAP pred_contribs",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return

    config = json.loads((args.cell / "source_manifest.json").read_text())["source_manifest"][
        "config"
    ]
    dataset_name = str(config["dataset"].get("name", ""))
    target_name = (
        "default risk"
        if dataset_name == "datafusion_default_2023"
        else "churn risk"
        if dataset_name == "rosbank"
        else "positive-class probability"
    )
    config["output"]["base_dir"] = str(args.cell)
    config.setdefault("input", {})["cot_features_base_dir"] = str(args.cell)
    packs = build_feature_sets(config, [args.baseline, args.claims])
    metrics = json.loads((args.cell / "ml_metrics.json").read_text(encoding="utf-8"))
    clusters = json.loads((args.cell / "cot_clusters.json").read_text(encoding="utf-8"))
    medoids = {
        str(row["feature"]): readable_claim(row["medoid"]) for row in clusters["cluster_meta"]
    }

    fitted = {}
    shap_values = {}
    probabilities = {}
    labels_by_set = {}
    ids_by_set = {}
    performance = {}
    for feature_set in (args.baseline, args.claims):
        pack = packs[feature_set]
        train_x, train_y = split_xy(pack["train"], pack["columns"])
        test_x, test_y = split_xy(pack["test"], pack["columns"])
        params = metrics[feature_set]["xgboost"]["params"]
        model = fit_model(config, params, train_x, train_y, seed=args.seed)
        fitted[feature_set] = model
        shap_values[feature_set] = tree_shap(model, test_x)
        probabilities[feature_set] = model.predict_proba(test_x)[:, 1]
        labels_by_set[feature_set] = test_y
        ids_by_set[feature_set] = pack["test"]["customer_id"].astype(str).tolist()
        performance[feature_set] = metrics[feature_set]["xgboost"]["runs"][str(args.seed)]["test"]

    if ids_by_set[args.baseline] != ids_by_set[args.claims]:
        raise ValueError("Baseline and claim test clients are not identically ordered")
    truth = labels_by_set[args.baseline]
    local_index, selection_rule = choose_local_example(
        ids_by_set[args.baseline],
        truth,
        probabilities[args.baseline],
        probabilities[args.claims],
        shap_values[args.baseline],
        shap_values[args.claims],
    )

    baseline_names = [raw_label(name) for name in packs[args.baseline]["columns"]]
    claim_names = [
        medoids.get(name, readable_claim(name)) for name in packs[args.claims]["columns"]
    ]
    left_top = top_indices(shap_values[args.baseline], args.top)
    right_top = top_indices(shap_values[args.claims], args.top)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 11.5))
    figure.subplots_adjust(
        left=0.22,
        right=0.985,
        top=0.88,
        bottom=0.105,
        wspace=0.66,
        hspace=0.42,
    )
    figure.patch.set_facecolor("#FBFAF7")
    for axis in axes.ravel():
        axis.set_facecolor("#FBFAF7")
    contribution_panel(
        axes[0, 0],
        shap_values[args.baseline],
        baseline_names,
        left_top,
        title=f"A  Conventional aggregate measurements ({len(baseline_names):,} inputs)",
        color="#7A869A",
    )
    contribution_panel(
        axes[0, 1],
        shap_values[args.claims],
        claim_names,
        right_top,
        title=f"B  Semantic behavioral concepts ({len(claim_names):,} inputs)",
        color="#2A9D8F",
    )
    client_id = ids_by_set[args.baseline][local_index]
    local_panel(
        axes[1, 0],
        shap_values[args.baseline][local_index],
        baseline_names,
        title=f"C  Pre-specified median-fidelity example · client {client_id}",
        target_name=target_name,
    )
    local_panel(
        axes[1, 1],
        shap_values[args.claims][local_index],
        claim_names,
        title="D  Same client, behavioral concepts",
        target_name=target_name,
    )
    figure.suptitle(
        "From low-level measurements to behavioral concepts for feature attribution",
        fontsize=17,
        fontweight="bold",
        x=0.5,
        y=0.975,
    )
    figure.text(
        0.5,
        0.943,
        "Same dataset, split, XGBoost family and TreeSHAP method · attribution describes model associations, not causal effects",
        ha="center",
        fontsize=10.5,
        color="#51555C",
    )
    figure.text(
        0.5,
        0.905,
        "Global panels rank features within each model by mean |SHAP|; magnitudes must not be compared across the two independently trained models.",
        ha="center",
        fontsize=9.5,
        color="#6A6E74",
    )
    figure.text(
        0.5,
        0.025,
        f"Local panels are signed: red raises and blue lowers predicted {target_name} log-odds. "
        "Claim concepts remain traceable to their cluster medoids and source atomic claims. "
        f"Test ROC–AUC: aggregate {performance[args.baseline].get('roc_auc', float('nan')):.3f}, "
        f"claims {performance[args.claims].get('roc_auc', float('nan')):.3f}.",
        ha="center",
        fontsize=9.5,
        color="#51555C",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            args.output.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
    plt.close(figure)
    atomic_write_json(
        args.output.with_suffix(".json"),
        {
            **plan,
            "client_id": client_id,
            "client_test_index": local_index,
            "client_selection": selection_rule,
            "truth": int(truth[local_index]),
            "dataset": dataset_name,
            "target_name": target_name,
            "baseline_probability": float(probabilities[args.baseline][local_index]),
            "claims_probability": float(probabilities[args.claims][local_index]),
            "performance": performance,
            "baseline_top_features": [packs[args.baseline]["columns"][i] for i in left_top[::-1]],
            "claim_top_features": [
                {"feature": packs[args.claims]["columns"][i], "medoid": claim_names[i]}
                for i in right_top[::-1]
            ],
        },
    )


if __name__ == "__main__":
    main()

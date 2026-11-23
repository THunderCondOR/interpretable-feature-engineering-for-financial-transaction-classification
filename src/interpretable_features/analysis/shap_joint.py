"""Compare conventional and semantic TreeSHAP blocks inside one model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

from interpretable_features.analysis.shap_comparison import raw_label, readable_claim
from interpretable_features.experiments.artifacts import atomic_write_json
from interpretable_features.models.ml_baseline import build_feature_sets, split_xy


def select_typical_client(ids, truth, probability, conventional, claims):
    prediction = (probability >= 0.5).astype(int)
    confidence = np.maximum(probability, 1 - probability)

    def signed(values):
        return (values > 1e-8).any(1) & (values < -1e-8).any(1)

    eligible = (prediction == truth) & (confidence >= 0.8) & signed(conventional) & signed(claims)
    pool = np.flatnonzero(eligible)
    fallback = False
    if not len(pool):
        pool = np.flatnonzero(prediction == truth)
        fallback = True
    median = float(np.median(confidence[pool]))
    index = min(pool.tolist(), key=lambda i: (abs(float(confidence[i]) - median), ids[i]))
    return index, {
        "rule": "correct prediction; confidence >= 0.80; both feature blocks have signed contributions; confidence nearest eligible median",
        "eligible_clients": int(len(pool)),
        "median_confidence": median,
        "selected_confidence": float(confidence[index]),
        "fallback_used": fallback,
    }


def global_panel(ax, values, labels, title, color, count=9):
    importance = np.abs(values).mean(0)
    indices = np.argsort(importance)[-min(count, len(labels)) :]
    ax.barh(range(len(indices)), importance[indices], color=color, alpha=0.92)
    ax.set_yticks(range(len(indices)), [labels[i] for i in indices])
    ax.set_title(title, loc="left", weight="bold", pad=12)
    ax.set_xlabel("mean |TreeSHAP value|")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return float(importance[indices].max())


def local_panel(ax, row, labels, title, target, count=7):
    positive = np.flatnonzero(row > 1e-8)
    negative = np.flatnonzero(row < -1e-8)
    n_neg = min(3, len(negative))
    n_pos = min(count - n_neg, len(positive))
    neg = negative[np.argsort(np.abs(row[negative]))[-n_neg:]] if n_neg else np.array([], int)
    pos = positive[np.argsort(np.abs(row[positive]))[-n_pos:]] if n_pos else np.array([], int)
    indices = np.asarray(sorted([*neg.tolist(), *pos.tolist()], key=lambda i: row[i]))
    values = row[indices]
    ax.barh(
        range(len(indices)), values, color=np.where(values >= 0, "#E45756", "#4C78A8"), alpha=0.92
    )
    ax.set_yticks(range(len(indices)), [labels[i] for i in indices])
    ax.axvline(0, color="#31343A", lw=0.8)
    ax.set_title(title, loc="left", weight="bold", pad=12)
    ax.set_xlabel(f"TreeSHAP contribution to {target} log-odds")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return float(np.abs(values).max()) if len(values) else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--feature-set", default="all_features")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {"cell": str(args.cell), "feature_set": args.feature_set, "output": str(args.output)},
            indent=2,
        )
    )
    if not args.execute:
        return

    config = json.loads((args.cell / "source_manifest.json").read_text())["source_manifest"][
        "config"
    ]
    config["output"]["base_dir"] = str(args.cell)
    config.setdefault("input", {})["cot_features_base_dir"] = str(args.cell)
    pack = build_feature_sets(config, [args.feature_set])[args.feature_set]
    x_train, y_train = split_xy(pack["train"], pack["columns"])
    x_test, y_test = split_xy(pack["test"], pack["columns"])
    metrics = json.loads((args.cell / "ml_metrics.json").read_text())
    params = metrics[args.feature_set]["xgboost"]["params"]
    model = XGBClassifier(
        **params,
        objective="binary:logistic",
        random_state=args.seed,
        n_jobs=4,
        tree_method="hist",
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    shap = model.get_booster().predict(xgb.DMatrix(x_test), pred_contribs=True)[:, :-1]

    names = pack["columns"]
    conventional_indices = np.asarray(
        [i for i, name in enumerate(names) if not name.startswith("claims__")]
    )
    claim_indices = np.asarray([i for i, name in enumerate(names) if name.startswith("claims__")])
    if not len(conventional_indices) or not len(claim_indices):
        raise ValueError("Joint model must contain conventional and claims__ feature blocks")
    clusters = json.loads((args.cell / "cot_clusters.json").read_text())
    medoids = {row["feature"]: row["medoid"] for row in clusters["cluster_meta"]}
    conventional_labels = [raw_label(names[i].split("__", 1)[-1], 38) for i in conventional_indices]
    claim_labels = [
        readable_claim(medoids.get(names[i].removeprefix("claims__"), names[i]), 46)
        for i in claim_indices
    ]
    conventional_shap, claim_shap = shap[:, conventional_indices], shap[:, claim_indices]
    ids = pack["test"]["customer_id"].astype(str).tolist()
    local, selection = select_typical_client(
        ids, y_test, probability, conventional_shap, claim_shap
    )
    dataset = str(config["dataset"].get("name"))
    target = "default risk" if dataset == "datafusion_default_2023" else "churn risk"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.5))
    fig.subplots_adjust(left=0.22, right=0.985, top=0.87, bottom=0.105, wspace=0.67, hspace=0.43)
    fig.patch.set_facecolor("#FBFAF7")
    for ax in axes.ravel():
        ax.set_facecolor("#FBFAF7")
    g1 = global_panel(
        axes[0, 0],
        conventional_shap,
        conventional_labels,
        f"A  Low-level measurements ({len(conventional_indices):,} inputs)",
        "#7A869A",
    )
    g2 = global_panel(
        axes[0, 1],
        claim_shap,
        claim_labels,
        f"B  Behavioral concepts ({len(claim_indices):,} inputs)",
        "#2A9D8F",
    )
    shared_global = max(g1, g2) * 1.06
    axes[0, 0].set_xlim(0, shared_global)
    axes[0, 1].set_xlim(0, shared_global)
    l1 = local_panel(
        axes[1, 0],
        conventional_shap[local],
        conventional_labels,
        f"C  Typical confident prediction · client {ids[local]}",
        target,
    )
    l2 = local_panel(
        axes[1, 1],
        claim_shap[local],
        claim_labels,
        "D  Same prediction, concept-level attribution",
        target,
    )
    shared_local = max(l1, l2) * 1.08
    axes[1, 0].set_xlim(-shared_local, shared_local)
    axes[1, 1].set_xlim(-shared_local, shared_local)
    fig.suptitle(
        "One model, one SHAP scale, two feature languages", fontsize=18, weight="bold", y=0.975
    )
    fig.text(
        0.5,
        0.942,
        "Conventional and semantic contributions come from the same claim-augmented XGBoost prediction; SHAP describes associations, not causal effects.",
        ha="center",
        color="#51555C",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.90,
        "Low-level measurements are individually readable but fragmented; behavioral concepts combine activity, magnitude, timing, and category context and remain traceable to atomic claims.",
        ha="center",
        color="#666A70",
        fontsize=9.5,
    )
    auc = metrics[args.feature_set]["xgboost"]["runs"][str(args.seed)]["test"].get("roc_auc")
    fig.text(
        0.5,
        0.025,
        f"Shared TreeSHAP scale. Red raises and blue lowers predicted {target} log-odds. Test ROC–AUC of the joint model: {auc:.3f}.",
        ha="center",
        color="#51555C",
        fontsize=9.5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)
    atomic_write_json(
        args.output.with_suffix(".json"),
        {
            "dataset": dataset,
            "cell": str(args.cell),
            "feature_set": args.feature_set,
            "method": "single-model XGBoost TreeSHAP pred_contribs",
            "seed": args.seed,
            "conventional_features": len(conventional_indices),
            "claim_features": len(claim_indices),
            "client_id": ids[local],
            "client_selection": selection,
            "truth": int(y_test[local]),
            "probability": float(probability[local]),
            "test_roc_auc": auc,
        },
    )


if __name__ == "__main__":
    main()

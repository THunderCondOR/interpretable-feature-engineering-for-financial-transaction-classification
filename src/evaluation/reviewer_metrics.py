"""Pure metrics for stability, grounding, and surrogate fidelity."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

GROUNDING_VERDICTS = (
    "supported", "partially_supported", "unsupported", "not_verifiable", "parse_error",
)


def clustering_agreement(assignments: dict[str, np.ndarray], axis: str) -> pd.DataFrame:
    """Pairwise ARI/NMI within one declared sensitivity axis."""
    rows = []
    for (left_name, left), (right_name, right) in combinations(assignments.items(), 2):
        if len(left) != len(right):
            raise ValueError("Assignments must use the same anchor set")
        rows.append({
            "axis": axis, "left": left_name, "right": right_name, "n_anchors": len(left),
            "ari": float(adjusted_rand_score(left, right)),
            "nmi": float(normalized_mutual_info_score(left, right)),
        })
    return pd.DataFrame(rows)


def cross_model_matching(left, right, left_weights=None, right_weights=None):
    """One-to-one and mutual-nearest cluster correspondence metrics."""
    similarity = cosine_similarity(np.asarray(left), np.asarray(right))
    rows, cols = linear_sum_assignment(-similarity)
    left_best, right_best = similarity.argmax(1), similarity.argmax(0)
    mutual = {(i, int(left_best[i])) for i in range(len(left)) if right_best[left_best[i]] == i}
    left_weights = np.ones(len(left)) if left_weights is None else np.asarray(left_weights, dtype=float)
    right_weights = np.ones(len(right)) if right_weights is None else np.asarray(right_weights, dtype=float)
    left_weights /= max(left_weights.sum(), 1.0)
    right_weights /= max(right_weights.sum(), 1.0)
    matched_left = float(left_weights[rows].sum())
    matched_right = float(right_weights[cols].sum())
    return {
        "one_to_one_mean_cosine": float(similarity[rows, cols].mean()),
        "mutual_nearest_pairs": int(len(mutual)),
        "mutual_nearest_share_left": float(len(mutual) / max(len(left), 1)),
        "size_weighted_left_best_cosine": float(np.average(similarity.max(1), weights=left_weights)),
        "size_weighted_right_best_cosine": float(np.average(similarity.max(0), weights=right_weights)),
        "matched_left_mass": matched_left, "matched_right_mass": matched_right,
        "unmatched_left_mass": 1.0 - matched_left, "unmatched_right_mass": 1.0 - matched_right,
        "pairs": [
            {"left": int(i), "right": int(j), "cosine": float(similarity[i, j]), "mutual_nearest": (int(i), int(j)) in mutual}
            for i, j in zip(rows, cols)
        ],
    }


def adjudicate_grounding(verdicts: list[str]) -> tuple[str, bool]:
    """Never manufacture a majority from a two-judge disagreement."""
    valid = [verdict if verdict in GROUNDING_VERDICTS else "parse_error" for verdict in verdicts]
    if not valid:
        return "missing", False
    counts = Counter(valid).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "disagreement", False
    return counts[0][0], True


def grounding_summary(records: pd.DataFrame, *, bootstrap_samples=1000, seed=17):
    """Return item verdicts, judge agreement, and client-level bootstrap CIs."""
    items = []
    for sample_id, group in records.groupby("sample_id", sort=False):
        raw_verdicts = [
            value if value in GROUNDING_VERDICTS else "parse_error"
            for value in group["verdict"].fillna("parse_error").tolist()
        ]
        verdict, majority = adjudicate_grounding(raw_verdicts)
        first = group.iloc[0]
        items.append({
            "sample_id": sample_id,
            "customer_id": first["customer_id"],
            "verdict": verdict,
            "has_majority": majority,
            "strict_consensus": len(set(raw_verdicts)) == 1,
            "has_disagreement": len(set(raw_verdicts)) > 1,
        })
    item_frame = pd.DataFrame(items)
    judges = records["judge_name"].dropna().unique().tolist()
    kappa = None
    if len(judges) == 2:
        pivot = records.pivot_table(index="sample_id", columns="judge_name", values="verdict", aggfunc="first").dropna()
        if len(pivot):
            kappa = float(cohen_kappa_score(pivot.iloc[:, 0], pivot.iloc[:, 1], labels=list(GROUNDING_VERDICTS)))
    rng, clients = np.random.default_rng(seed), item_frame["customer_id"].unique()
    proportions = {verdict: [] for verdict in (*GROUNDING_VERDICTS, "disagreement")}
    if len(clients):
        by_client = {cid: item_frame[item_frame["customer_id"] == cid] for cid in clients}
        for _ in range(bootstrap_samples):
            sampled = rng.choice(clients, len(clients), replace=True)
            boot = pd.concat([by_client[cid] for cid in sampled], ignore_index=True)
            for verdict in proportions:
                proportions[verdict].append(float((boot["verdict"] == verdict).mean()))
    summary = {
        "n_items": int(len(item_frame)), "n_clients": int(len(clients)), "cohen_kappa": kappa,
        "strict_consensus_share": float(item_frame["strict_consensus"].mean()) if len(item_frame) else 0.0,
        "disagreement_share": float(item_frame["has_disagreement"].mean()) if len(item_frame) else 0.0,
        "verdicts": {},
    }
    for verdict, samples in proportions.items():
        estimate = float((item_frame["verdict"] == verdict).mean()) if len(item_frame) else 0.0
        summary["verdicts"][verdict] = {
            "share": estimate,
            "ci_low": float(np.quantile(samples, 0.025)) if samples else None,
            "ci_high": float(np.quantile(samples, 0.975)) if samples else None,
        }
    return item_frame, summary


def seed_summary(rows: list[dict], metric_names: list[str]) -> dict:
    """Keep one summary per experimental cell; never pool granularity axes."""
    result = {"n_seeds": len(rows), "seeds": [row.get("seed") for row in rows]}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()),
        }
    return result


def paired_client_bootstrap_delta(
    frame: pd.DataFrame,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    prediction_a="prediction_a",
    prediction_b="prediction_b",
    truth="label",
    client="customer_id",
    samples=2000,
    seed=17,
):
    """Paired CI for metric(A)-metric(B), resampling clients."""
    clients = frame[client].unique()
    rng, deltas = np.random.default_rng(seed), []
    groups = {cid: frame[frame[client] == cid] for cid in clients}
    for _ in range(samples):
        sampled = rng.choice(clients, len(clients), replace=True)
        boot = pd.concat([groups[cid] for cid in sampled], ignore_index=True)
        deltas.append(metric(boot[truth], boot[prediction_a]) - metric(boot[truth], boot[prediction_b]))
    point = metric(frame[truth], frame[prediction_a]) - metric(frame[truth], frame[prediction_b])
    return {"delta": float(point), "ci_low": float(np.quantile(deltas, 0.025)), "ci_high": float(np.quantile(deltas, 0.975))}


def surrogate_fidelity(teacher_probabilities, surrogate_probabilities, labels):
    teacher = np.asarray(teacher_probabilities, dtype=float)
    surrogate = np.asarray(surrogate_probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    teacher = np.clip(teacher, 1e-12, 1.0)
    surrogate = np.clip(surrogate, 1e-12, 1.0)
    teacher /= teacher.sum(1, keepdims=True)
    surrogate /= surrogate.sum(1, keepdims=True)
    teacher_pred, surrogate_pred = teacher.argmax(1), surrogate.argmax(1)
    midpoint = 0.5 * (teacher + surrogate)
    js = 0.5 * (
        np.sum(teacher * np.log(teacher / midpoint), axis=1)
        + np.sum(surrogate * np.log(surrogate / midpoint), axis=1)
    )
    teacher_correct, surrogate_correct = teacher_pred == labels, surrogate_pred == labels
    result = {
        "hard_agreement": float((teacher_pred == surrogate_pred).mean()),
        "probability_mae": float(np.abs(teacher - surrogate).mean()),
        "probability_rmse": float(np.sqrt(np.square(teacher - surrogate).mean())),
        "jensen_shannon_divergence": float(js.mean()),
        "agree_and_correct": float(((teacher_pred == surrogate_pred) & teacher_correct).mean()),
        "agree_and_wrong": float(((teacher_pred == surrogate_pred) & ~teacher_correct).mean()),
        "teacher_only_correct": float((teacher_correct & ~surrogate_correct).mean()),
        "surrogate_only_correct": float((~teacher_correct & surrogate_correct).mean()),
    }
    confidence = teacher.max(axis=1)
    result["confidence_coverage"] = []
    for threshold in np.linspace(0.5, 0.95, 10):
        selected = confidence >= threshold
        result["confidence_coverage"].append({
            "threshold": float(threshold),
            "coverage": float(selected.mean()),
            "agreement": (
                float((teacher_pred[selected] == surrogate_pred[selected]).mean())
                if selected.any()
                else None
            ),
        })
    return result


def cluster_occlusion(predict_proba: Callable[[np.ndarray], np.ndarray], x: np.ndarray, shown_indices: list[int]):
    """Classifier-level occlusion, comprehensiveness, and sufficiency."""
    row = np.asarray(x, dtype=float).reshape(1, -1)
    base = predict_proba(row)[0]
    predicted = int(base.argmax())
    per_cluster = {}
    for index in shown_indices:
        masked = row.copy()
        masked[0, index] = 0
        per_cluster[str(index)] = float(base[predicted] - predict_proba(masked)[0, predicted])
    removed = row.copy()
    removed[0, shown_indices] = 0
    only = np.zeros_like(row)
    only[0, shown_indices] = row[0, shown_indices]
    return {
        "predicted_class": predicted,
        "base_probability": float(base[predicted]),
        "per_cluster_occlusion": per_cluster,
        "comprehensiveness": float(base[predicted] - predict_proba(removed)[0, predicted]),
        "sufficiency_probability": float(predict_proba(only)[0, predicted]),
    }

"""Self-contained interactive report and publication exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from interpretable_features.experiments.artifacts import fingerprint

TABS = [
    "Run status",
    "Metrics",
    "Reference deltas",
    "Cluster map",
    "Cluster table",
    "Cross-model matching",
    "Stability",
    "Classifier/tree explorer",
    "Fidelity",
    "Grounding",
    "Provenance",
]


def synthetic_bundle(seed=17):
    rng = np.random.default_rng(seed)
    claims, clusters = [], []
    themes = ["Everyday spending", "Mobility", "Digital services", "Cashflow"]
    colors = ["#58b368", "#f39c35", "#4f86c6", "#af7ac5"]
    for index in range(20):
        theme, cluster_id = index % 4, f"clu_{index:03d}"
        center = np.array([theme * 4.2 + rng.normal(0, 0.5), (index // 4) * 2 + rng.normal(0, 0.4)])
        size = int(rng.integers(18, 60))
        clusters.append(
            {
                "cluster_id": cluster_id,
                "macro_theme": themes[theme],
                "color": colors[theme],
                "medoid": f"Representative behavioural claim {index}",
                "occurrences": size * 2,
                "unique_clients": size,
                "compactness": float(rng.uniform(0.05, 0.28)),
                "prevalence": float(rng.uniform(0.02, 0.22)),
                "lift": float(rng.uniform(-0.8, 1.4)),
                "cross_model_match": f"gpt_clu_{index:03d}",
                "match_cosine": float(rng.uniform(0.62, 0.96)),
                "tree_nodes": [int(rng.integers(1, 15))],
                "importance": float(rng.uniform(0, 0.12)),
            }
        )
        for occurrence in range(size):
            point = center + rng.normal(0, [0.55, 0.42])
            claims.append(
                {
                    "claim_id": f"claim_{index:03d}_{occurrence:03d}",
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "cluster_id": cluster_id,
                    "macro_theme": themes[theme],
                    "color": colors[theme],
                    "dataset": ["gender", "age", "rosbank"][index % 3],
                    "model": ["qwen", "gpt_oss"][index % 2],
                    "variant": "demo",
                    "seed": [17, 101, 947][index % 3],
                    "text": f"Synthetic claim {occurrence} in cluster {index}",
                    "customer_id": index * 1000 + occurrence,
                    "label": occurrence % 2,
                    "distance": float(rng.uniform(0, 0.45)),
                }
            )
    clusters.append(
        {
            "cluster_id": "unassigned",
            "macro_theme": "Unassigned",
            "color": "#b8bcc5",
            "medoid": "Semantic outliers / unassigned claims",
            "occurrences": 30,
            "unique_clients": 26,
            "compactness": 1.0,
            "prevalence": 0.01,
            "lift": 0.0,
            "cross_model_match": "none",
            "match_cosine": 0.0,
            "tree_nodes": [],
            "importance": 0.0,
        }
    )
    for occurrence in range(30):
        claims.append(
            {
                "claim_id": f"outlier_{occurrence:03d}",
                "x": float(rng.uniform(-1, 14)),
                "y": float(rng.uniform(-1, 10)),
                "cluster_id": "unassigned",
                "macro_theme": "Unassigned",
                "color": "#b8bcc5",
                "dataset": ["gender", "age", "rosbank"][occurrence % 3],
                "model": ["qwen", "gpt_oss"][occurrence % 2],
                "variant": "demo",
                "seed": [17, 101, 947][occurrence % 3],
                "text": f"Synthetic unassigned claim {occurrence}",
                "customer_id": 90000 + occurrence,
                "label": occurrence % 2,
                "distance": 1.0,
            }
        )
    metrics = [
        {
            "dataset": dataset,
            "model": model,
            "experiment": experiment,
            "balanced_accuracy": float(rng.uniform(0.62, 0.87)),
            "sd": float(rng.uniform(0.002, 0.018)),
            "article_delta": float(rng.uniform(-0.03, 0.05)),
        }
        for dataset in ["gender", "age", "rosbank"]
        for model in ["qwen", "gpt_oss"]
        for experiment in ["standard", "handcrafted", "cot", "concat"]
    ]
    return {
        "synthetic": True,
        "claims": claims,
        "clusters": clusters,
        "metrics": metrics,
        "stability": {
            "axes": ["generation_seed", "clustering_seed", "threshold", "fixed_k", "coverage"],
            "ari_mean": 0.78,
            "nmi_mean": 0.82,
        },
        "grounding": {
            "supported": 0.72,
            "partially_supported": 0.16,
            "unsupported": 0.07,
            "not_verifiable": 0.05,
            "kappa": 0.67,
        },
        "fidelity": {"hard_agreement": 0.81, "probability_mae": 0.09, "jensen_shannon": 0.035},
        "cross_model": {
            "mutual_nearest_share": 0.58,
            "weighted_cosine": 0.79,
            "unmatched_mass": 0.14,
        },
        "tree_paths": [],
        "provenance": {"mode": "synthetic fixture", "note": "No empirical result is represented."},
    }


def _empty_bundle(run_id: str, warning: str | None = None):
    provenance = {"run_id": run_id, "mode": "empirical artifacts"}
    if warning:
        provenance["warning"] = warning
    return {
        "synthetic": False,
        "claims": [],
        "clusters": [],
        "metrics": [],
        "stability": {},
        "grounding": {},
        "fidelity": {},
        "cross_model": {},
        "tree_paths": [],
        "provenance": provenance,
    }


def _cluster_artifacts(root: Path, run_id: str, dataset: str, model: str):
    path = root / "cot_clusters.json"
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cluster_meta", []) if isinstance(payload, dict) else payload
    centroids_path = root / "cot_cluster_model.npz"
    coordinates = np.zeros((len(rows), 2), dtype=float)
    if centroids_path.exists() and rows:
        centroids = np.load(centroids_path)["centroids"]
        if len(centroids) == len(rows):
            if min(centroids.shape) >= 2:
                from sklearn.decomposition import PCA

                coordinates = PCA(n_components=2, random_state=17).fit_transform(centroids)
            elif centroids.shape[1] == 1:
                coordinates[:, 0] = centroids[:, 0]
    palette = ["#58b368", "#f39c35", "#4f86c6", "#af7ac5", "#ef6f6c"]
    clusters, claims = [], []
    total_clients = max(sum(int(row.get("unique_clients", 0)) for row in rows), 1)
    for index, row in enumerate(rows):
        source_cluster_id = str(
            row.get("cluster_id") or row.get("feature") or f"cluster_{index:04d}"
        )
        cluster_id = f"{run_id}::{dataset}::{model}::{source_cluster_id}"
        medoid = str(row.get("medoid") or (row.get("examples") or [source_cluster_id])[0])
        clients = int(row.get("unique_clients", row.get("size", 0)))
        color = palette[index % len(palette)]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "source_cluster_id": source_cluster_id,
                "run_id": run_id,
                "macro_theme": str(row.get("macro_theme", "Unassigned theme")),
                "color": color,
                "medoid": medoid,
                "occurrences": int(row.get("occurrences", row.get("size", 0))),
                "unique_clients": clients,
                "compactness": float(
                    row.get("compactness_mean_distance", row.get("compactness", 0.0))
                ),
                "prevalence": float(row.get("prevalence", clients / total_clients)),
                "lift": float(row.get("lift", 0.0)),
                "cross_model_match": str(row.get("cross_model_match", "")),
                "match_cosine": float(row.get("match_cosine", 0.0)),
                "tree_nodes": list(row.get("tree_nodes", [])),
                "importance": float(row.get("importance", 0.0)),
                "dataset": dataset,
                "model": model,
            }
        )
        claims.append(
            {
                "claim_id": f"{dataset}:{model}:{cluster_id}",
                "x": float(coordinates[index, 0]),
                "y": float(coordinates[index, 1]),
                "cluster_id": cluster_id,
                "macro_theme": str(row.get("macro_theme", "Unassigned theme")),
                "color": color,
                "dataset": dataset,
                "model": model,
                "variant": "empirical",
                "seed": 0,
                "text": medoid,
                "customer_id": -1,
                "label": -1,
                "distance": float(row.get("compactness_mean_distance", 0.0)),
                "point_kind": "train_cluster_centroid",
            }
        )
    return clusters, claims


def _metric_artifacts(root: Path, dataset: str, model: str):
    rows = []
    ml_path = root / "ml_metrics.json"
    if ml_path.exists():
        payload = json.loads(ml_path.read_text(encoding="utf-8"))
        for experiment, result in payload.items():
            if not isinstance(result, dict):
                continue
            for classifier in ("xgboost", "decision_tree"):
                classifier_result = result.get(classifier, {})
                metrics = classifier_result.get("test", {})
                if metrics:
                    seeded = (
                        classifier_result.get("summary", {})
                        .get("test", {})
                        .get("balanced_accuracy", {})
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "experiment": f"{experiment}/{classifier}",
                            "balanced_accuracy": seeded.get(
                                "mean", metrics.get("balanced_accuracy")
                            ),
                            "sd": seeded.get("sd"),
                            "n_seeds": len(classifier_result.get("runs", {})) or None,
                            "article_delta": None,
                        }
                    )
    for path in root.glob("llm_metrics_*.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("balanced_accuracy") is not None:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "experiment": f"llm_direct/{metrics.get('split', path.stem)}",
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "sd": None,
                    "article_delta": None,
                }
            )
    return rows


def load_bundle(results_root: Path, run_id: str, *, allow_missing=False, strict=False):
    if allow_missing and strict:
        raise ValueError("strict and allow_missing are mutually exclusive")
    manifests = sorted(
        {
            *results_root.glob("**/manifest.json"),
            *results_root.glob("**/run_manifest.json"),
        }
    )
    selected = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest_run_id = payload.get("run_id") or payload.get("experiment", {}).get("run_id")
        if manifest_run_id == run_id:
            selected.append((path, payload))
    if not selected:
        if not allow_missing:
            raise FileNotFoundError(f"No manifests for run_id={run_id} below {results_root}")
        return _empty_bundle(
            run_id, "No compatible manifests found; no numerical placeholders were fabricated."
        )
    stability_rows = []
    fidelity_payloads = []
    tree_paths = []

    if strict:
        invalid = []
        for path, payload in selected:
            identity = {
                "manifest_version": payload.get("manifest_version"),
                "config_sha256": payload.get("config_sha256"),
                "dataset_files": payload.get("dataset_files"),
                "prompt_files": payload.get("prompt_files"),
                "git_revision": payload.get("git_revision"),
                "packages": payload.get("runtime", {}).get("packages"),
            }
            if payload.get("manifest_version", 0) < 2 or payload.get(
                "manifest_sha256"
            ) != fingerprint(identity):
                invalid.append(str(path))
        if invalid:
            raise ValueError(f"Incompatible manifests: {invalid}")

    bundle = _empty_bundle(run_id)
    artifact_roots = []
    for path, manifest in selected:
        root = path.parent
        artifact_roots.append(str(root))
        config = manifest.get("config", {})
        dataset = str(config.get("dataset", {}).get("name", "unknown"))
        model = str(
            manifest.get("model_id")
            or config.get("experiment", {}).get("model_slug")
            or config.get("llm", {}).get("default_model", "unknown")
        )
        clusters, claims = _cluster_artifacts(root, run_id, dataset, model)
        bundle["clusters"].extend(clusters)
        bundle["claims"].extend(claims)
        bundle["metrics"].extend(_metric_artifacts(root, dataset, model))
        stability_path = root / "cluster_stability" / "summary.json"
        if stability_path.exists():
            payload = json.loads(stability_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                stability_rows.extend(
                    {"dataset": dataset, "model": model, **row}
                    for row in payload
                    if isinstance(row, dict)
                )
        for fidelity_path in root.glob("**/fidelity_metrics.json"):
            fidelity_payloads.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "path": str(fidelity_path),
                    "payload": json.loads(fidelity_path.read_text(encoding="utf-8")),
                }
            )
        for tree_path in root.glob("**/tree_decision_paths.jsonl"):
            tree_paths.extend(
                json.loads(line)
                for line in tree_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

    if stability_rows:
        groups = []
        frame = pd.DataFrame(stability_rows)
        for keys, group in frame.groupby(["dataset", "model", "axis"], dropna=False):
            item = {"dataset": keys[0], "model": keys[1], "axis": keys[2], "n": len(group)}
            for output, column in (
                ("ari", "train_assignment_ari_vs_reference"),
                ("nmi", "train_assignment_nmi_vs_reference"),
            ):
                values = (
                    pd.to_numeric(group[column], errors="coerce").dropna()
                    if column in group
                    else pd.Series(dtype=float)
                )
                item[f"{output}_mean"] = float(values.mean()) if len(values) else None
                item[f"{output}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            groups.append(item)
        bundle["stability"] = {"groups": groups, "rows": stability_rows}
    bundle["tree_paths"] = tree_paths
    if fidelity_payloads:
        groups = []
        for artifact in fidelity_payloads:
            for classifier, result in artifact["payload"].get("results", {}).items():
                metrics = result.get("test", {}) if isinstance(result, dict) else {}
                if metrics:
                    groups.append(
                        {
                            "dataset": artifact["dataset"],
                            "model": artifact["model"],
                            "classifier": classifier,
                            "artifact": artifact["path"],
                            **{
                                key: value
                                for key, value in metrics.items()
                                if isinstance(value, (int, float))
                            },
                        }
                    )
        bundle["fidelity"] = {"groups": groups}

    grounding_path = results_root / "grounding" / run_id / "grounding.metrics.json"
    if grounding_path.exists():
        grounding_cells = json.loads(grounding_path.read_text(encoding="utf-8"))
        denominator = sum(int(cell.get("n_items", 0)) for cell in grounding_cells.values())
        bundle["grounding"] = {
            verdict: (
                sum(
                    int(cell.get("n_items", 0))
                    * float(cell.get("verdicts", {}).get(verdict, {}).get("share", 0.0))
                    for cell in grounding_cells.values()
                )
                / denominator
            )
            for verdict in (
                "supported",
                "partially_supported",
                "unsupported",
                "not_verifiable",
                "parse_error",
                "disagreement",
            )
            if denominator
        }

    cross_path = results_root / "analysis" / run_id / "cross_model_clusters.json"
    if cross_path.exists():
        cross_payload = json.loads(cross_path.read_text(encoding="utf-8"))
        summaries = [
            row.get("summary", {})
            for row in cross_payload.get("details", {}).values()
            if isinstance(row, dict)
        ]
        keys = (
            "one_to_one_mean_cosine",
            "mutual_nearest_share_left",
            "size_weighted_left_best_cosine",
            "size_weighted_right_best_cosine",
            "unmatched_left_mass",
            "unmatched_right_mass",
        )
        bundle["cross_model"] = {
            key: float(np.mean([row[key] for row in summaries if row.get(key) is not None]))
            for key in keys
            if any(row.get(key) is not None for row in summaries)
        }

    missing = []
    for name in ("clusters", "claims", "metrics"):
        if not bundle[name]:
            missing.append(name)
    bundle["provenance"].update(
        {
            "manifests": [str(path) for path, _ in selected],
            "artifact_roots": artifact_roots,
            "missing_sections": missing,
        }
    )
    if strict and missing:
        raise FileNotFoundError(f"Strict report is missing artifacts: {missing}")
    return bundle


def portable_bundle(bundle, max_points=3000):
    if len(bundle["claims"]) <= max_points:
        return bundle
    frame = pd.DataFrame(bundle["claims"])
    per = max(1, max_points // frame.cluster_id.nunique())
    sampled = pd.concat(
        [
            group.sample(min(len(group), per), random_state=17)
            for _, group in frame.groupby("cluster_id")
        ],
        ignore_index=True,
    )
    return {
        **bundle,
        "claims": sampled.to_dict(orient="records"),
        "portable_sampling": {"displayed": len(sampled), "total": len(frame)},
    }


def export_publication_map(bundle, output_prefix: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull, QhullError

    claims, clusters = pd.DataFrame(bundle["claims"]), pd.DataFrame(bundle["clusters"])
    fig, (ax, legend) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})
    if claims.empty or clusters.empty:
        ax.axis("off")
        legend.axis("off")
        ax.text(
            0.5,
            0.5,
            "Cluster artifacts are not available yet.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    fig.patch.set_facecolor("white")
    for cluster_id, group in claims.groupby("cluster_id"):
        meta = clusters[clusters.cluster_id == cluster_id].iloc[0]
        ax.scatter(
            group.x,
            group.y,
            s=10 + meta.prevalence * 110,
            c=meta.color,
            alpha=0.72,
            edgecolors="none",
        )
    for theme, group in claims.groupby("macro_theme"):
        if theme == "Unassigned":
            continue
        points = group[["x", "y"]].to_numpy()
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                polygon = points[hull.vertices]
            except QhullError:
                polygon = None
            if polygon is not None:
                ax.plot(*np.vstack([polygon, polygon[0]]).T, color="#23314d", lw=1.2, alpha=0.7)
            ax.text(
                points[:, 0].mean(),
                points[:, 1].max() + 0.3,
                theme,
                ha="center",
                fontsize=9,
                bbox=dict(fc="white", ec="none", alpha=0.8),
            )
    ax.set(xticks=[], yticks=[])
    ax.spines[:].set_visible(False)
    ax.set_title("Atomic claim clusters and macro-themes", loc="left", weight="bold")
    legend.axis("off")
    legend.set_title("Representative claims", loc="left", fontweight="bold")
    for index, row in (
        clusters.sort_values("prevalence", ascending=False)
        .head(7)
        .reset_index(drop=True)
        .iterrows()
    ):
        legend.scatter(0.03, 0.88 - index * 0.12, s=75, c=row.color, transform=legend.transAxes)
        legend.text(
            0.1,
            0.88 - index * 0.12,
            f"{index + 1}. {row.medoid}\n   clients={row.unique_clients} · lift={row.lift:+.2f}",
            va="center",
            fontsize=8,
            transform=legend.transAxes,
        )
    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output_prefix.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plotly_source():
    from plotly.offline import get_plotlyjs

    source = get_plotlyjs()
    # Plotly embeds a default topojson CDN URL even when no geo trace is used.
    # Remove it so the generated document contains no network endpoint at all.
    return source.replace("https://cdn.plot.ly/", "")


def build_report(bundle, run_id):
    from jinja2 import Environment, PackageLoader, select_autoescape

    template = Environment(
        loader=PackageLoader("interpretable_features", "reporting"),
        autoescape=select_autoescape(["html"]),
    ).get_template("report.html")
    records = bundle["claims"] + bundle["metrics"]
    return template.render(
        dataset_count=len({row["dataset"] for row in records if row.get("dataset")}),
        model_count=len({row["model"] for row in records if row.get("model")}),
        run_id=run_id,
        mode="SYNTHETIC DEMO" if bundle.get("synthetic") else "RESULTS / PLACEHOLDERS",
        tabs=TABS,
        plotly_js=_plotly_source(),
        data_json=json.dumps(bundle, ensure_ascii=False).replace("<", "\\u003c"),
    )
